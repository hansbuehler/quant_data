"""Daily quote containment and rolling in-sample reference quantiles."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from quantdata.dysanos.flex.run_all_history_joint_direct import _geometry_from_completion
from quantdata.dysanos.v11.jax_affine200 import load_quotes, make_functions
from quantdata.dysanos.v11.jax_geometry import with_model_business_dtes
from quantdata.dysanos.v11.run_rolling_direct_ar import _geometry


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("completion", "panel", "archive", "initialization", "checkpoint", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--model-business-dtes", type=int, nargs="+", required=True)
    parser.add_argument("--out-of-sample", type=Path)
    args = parser.parse_args()
    geometry = _geometry(_geometry_from_completion(json.loads(args.completion.read_text()), requested_model_business_dtes=None), lower=-4, upper=3)
    geometry = with_model_business_dtes(geometry, args.model_business_dtes, allow_zero_backbone=True)
    with np.load(args.initialization / "states.npz") as saved:
        dates = tuple(date.fromisoformat(str(day)) for day in saved["dates"])
        if not np.array_equal(saved["model_business_dtes"], geometry.model_business_dtes):
            raise ValueError("Initialization expiry grid mismatch")
    with np.load(args.checkpoint) as saved:
        params = {key: jnp.asarray(saved[key]) for key in ("b", "B", "h")}
        iteration = int(saved["iteration"])
    quotes = load_quotes(dates, args.panel, args.archive, geometry)
    _, price_fn = make_functions(geometry, "volatility")
    rows = []
    for offset in range(0, len(dates), 5):
        indices = np.arange(offset, min(offset + 5, len(dates)))
        block = {key: jnp.asarray(value[indices]) for key, value in quotes.items()}
        state = params["b"][None, :] + params["h"][indices] @ params["B"]
        price, _ = price_fn(state, block)
        inside = np.asarray(jnp.sum(block["mask"] & (price >= block["bids"]) & (price <= block["asks"]), axis=1))
        count = np.asarray(jnp.sum(block["mask"], axis=1))
        rows.extend({"date": str(dates[i]), "quotes": int(n), "inside_percent": float(100 * k / n)}
                    for i, n, k in zip(indices, count, inside, strict=True))
        if (offset + 5) % 50 == 0:
            print(f"days={offset + 5}/{len(dates)}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output, index=False)
    p10, p90 = float(frame.inside_percent.quantile(.1)), float(frame.inside_percent.quantile(.9))
    report = {"iteration": iteration, "days": len(frame), "first": rows[0], "last": rows[-1],
              "p10": p10, "p90": p90, "median": float(frame.inside_percent.median()),
              "weighted_inside_percent": float(np.average(frame.inside_percent, weights=frame.quotes))}
    if args.out_of_sample is not None:
        samples = json.loads(args.out_of_sample.read_text())
        report["out_of_sample"] = [{**row, "below_p10": row["inside_percent"] < p10,
                                    "above_p90": row["inside_percent"] > p90} for row in samples]
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
