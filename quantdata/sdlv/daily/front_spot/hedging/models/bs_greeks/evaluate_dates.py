"""Evaluate unit-payout Black Greek hedges on an existing comparison date panel."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from datetime import date
from pathlib import Path

from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    PublicHedgingPanel,
    default_public_candidate_path,
)
from quantdata.sdlv.daily.front_spot.hedging.models.bs_greeks.adapter import (
    run_bs_greeks_daily_book,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dates-csv", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=default_public_candidate_path())
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    with arguments.dates_csv.open(newline="") as stream:
        dates = tuple(sorted({date.fromisoformat(row["date"]) for row in csv.DictReader(stream)}))
    if not dates:
        raise ValueError("comparison CSV has no dates")
    panel = PublicHedgingPanel(arguments.dataset)
    identity = {
        "schema": "bs-greeks-unit-digital-comparison-v1",
        "dataset_manifest_sha256": hashlib.sha256(
            (arguments.dataset / "manifest.json").read_bytes()
        ).hexdigest(),
        "dates_csv_sha256": hashlib.sha256(arguments.dates_csv.read_bytes()).hexdigest(),
        "dates": [origin.isoformat() for origin in dates],
        "target_maturity_business_days": 1,
        "claim_quantity": 1.0,
    }
    arguments.output.mkdir(parents=True, exist_ok=True)
    run_path = arguments.output / "run.json"
    if run_path.exists():
        if json.loads(run_path.read_text()) != identity:
            raise ValueError("Black Greek comparison run identity changed")
    else:
        run_path.write_text(json.dumps(identity, indent=2) + "\n")
    records = arguments.output / "records"
    records.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    for index, origin in enumerate(dates, start=1):
        path = records / f"{origin.isoformat()}.json"
        if not path.exists():
            results = run_bs_greeks_daily_book(
                panel,
                origin_date=origin,
                target_maturity=1,
            )
            if any(not result.converged for result in results):
                raise RuntimeError(f"Black Greek hedge did not converge on {origin}")
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps({"results": [result.to_dict() for result in results]}, indent=2)
            )
            temporary.replace(path)
        if index % 10 == 0 or index == len(dates):
            elapsed = time.monotonic() - started
            eta = elapsed * (len(dates) - index) / index
            print(
                f"Black Greek dates {index}/{len(dates)} elapsed_s={elapsed:.1f} "
                f"eta_s={eta:.1f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
