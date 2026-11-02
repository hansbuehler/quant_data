"""Track saved validation bid/ask checkpoints against the unchanged Q expectation."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from quantdata.gromm2.gpu_forward_queue import resource_report


def selected_price_snapshot(report: dict, history: list[dict], expectation: dict) -> dict | None:
    """Combine separately selected policies, never the three latest epoch weights."""
    selected, best_epochs = {}, {}
    for name in ("zero", "short", "long"):
        best_epochs[name] = report["policies"][name]["best_epoch"]
        row = next((row for row in history if row["epoch"] == best_epochs[name]), None)
        if row is None:
            return None
        selected[name] = float(row["validation"][name]["DH"])
    bid = selected["zero"] - selected["long"]
    ask = selected["short"] - selected["zero"]
    mean = float(expectation["mean"])
    if not all(math.isfinite(value) for value in (bid, ask, mean)):
        raise ValueError("price monitoring requires finite risks and expectation")
    return dict(epoch=report["epoch"], training_stop_reason=report["stop_reason"],
        best_epochs=best_epochs, bid=bid, ask=ask, expectation=expectation,
        q_between_point_quotes=bid <= mean <= ask, crossed_point_quotes=bid > ask,
        bid_mc_standard_error=None, ask_mc_standard_error=None,
        cost_basis=report["config"].get("turnover_cost_basis", "legacy_spot_notional"),
        interpretation="Selected-checkpoint validation risks, with costs. Interim bid/ask MC errors are not recorded in epoch histories; a point crossing alone is not a significance test.")


def _write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def final_quotes_match_selection(quotes: dict, report: dict) -> bool:
    """Never overlay resumed training prices with an older completed valuation."""
    expected = {"epoch": report["epoch"],
        "best_epochs": {name: record["best_epoch"] for name, record in report["policies"].items()}}
    return quotes.get("selection") == expected


def monitor_prices(run_directory: Path, *, control_directory: Path, output_directory: Path,
                    case: str, kind: str, interval: float = 60.0, policy_directory: Path | None = None) -> None:
    """Read tiny saved reports only; never load scenarios or run neural evaluation."""
    if case not in {"black0", "black5"} or kind not in {"cohorts", "factors", "combined"} or not 0 < interval <= 60:
        raise ValueError("invalid monitored book or polling interval")
    output_directory.mkdir(parents=True, exist_ok=False)
    expectation = json.loads((run_directory / "validation_report.json").read_text())["prices"][case]
    previous = None
    while True:
        resources = resource_report()
        if resources["blocked"]:
            raise RuntimeError("price monitor stopped by resource health; no new evaluation launched")
        if resources["warning"]:
            print(json.dumps(dict(event="price-monitor-resource-warning", resources=resources)), flush=True)
        snapshot = None
        report_paths = ([policy_directory / "training_report.json"] if policy_directory is not None else
            sorted((run_directory / "hedges" / case / kind).glob("*/training_report.json")))
        for path in report_paths:
            if not path.exists():
                continue
            report = json.loads(path.read_text())
            if report["config"].get("turnover_cost_basis") != "unit_l1":
                continue
            history_path = path.with_name("training_history.json")
            if not history_path.exists():
                continue
            history = json.loads(history_path.read_text())["epochs"]
            candidate = selected_price_snapshot(report, history, expectation)
            if candidate is None:
                continue
            if snapshot is not None:
                raise ValueError("multiple unit-turnover runs for the same monitored book")
            snapshot = candidate | dict(policy_directory=str(path.parent))
            quote_path = path.with_name("quotes.json")
            quotes = json.loads(quote_path.read_text()) if quote_path.exists() else {}
            if final_quotes_match_selection(quotes, report):
                final = quotes["validation"]["statistics"]
                snapshot.update(bid=final["bid"]["mean"], ask=final["ask"]["mean"],
                    bid_mc_standard_error=final["bid"]["standard_error"],
                    ask_mc_standard_error=final["ask"]["standard_error"],
                    bid_three_se=[final["bid"]["three_se_lower"], final["bid"]["three_se_upper"]],
                    ask_three_se=[final["ask"]["three_se_lower"], final["ask"]["three_se_upper"]])
                snapshot.update(q_between_point_quotes=snapshot["bid"] <= expectation["mean"] <= snapshot["ask"],
                    crossed_point_quotes=snapshot["bid"] > snapshot["ask"])
        state = dict(case=case, kind=kind, units="inception spot", cost_basis="unit_l1", prices=snapshot,
            phase="tracking-best-validation-checkpoints" if snapshot else "waiting-for-first-saved-quotes")
        signature = json.dumps(state, sort_keys=True)
        if signature != previous:
            timestamp = time.time()
            record = state | dict(timestamp=timestamp)
            _write(output_directory / "latest.json", record)
            with (output_directory / "history.jsonl").open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            alert = snapshot is not None and (not snapshot["q_between_point_quotes"] or snapshot["crossed_point_quotes"])
            print(json.dumps(dict(event="price-monitor-point-alert" if alert else "price-monitor-update", **record)), flush=True)
            previous = signature
        queue_path = control_directory / "gpu_queue_status.json"
        queue = json.loads(queue_path.read_text()) if queue_path.exists() else {}
        if queue.get("event") == "allocation_released":
            print(json.dumps(dict(event="price-monitor-job-ended", child_returncode=queue.get("child_returncode"))), flush=True)
            return
        time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--control-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--case", choices=("black0", "black5"), required=True)
    parser.add_argument("--kind", choices=("cohorts", "factors", "combined"), required=True)
    parser.add_argument("--policy-directory", type=Path,
        help="Track a separate fresh-run checkpoint directory rather than the legacy default root")
    arguments = parser.parse_args()
    monitor_prices(arguments.run_directory, control_directory=arguments.control_directory,
        output_directory=arguments.output_directory, case=arguments.case, kind=arguments.kind,
        policy_directory=arguments.policy_directory)


if __name__ == "__main__":
    main()
