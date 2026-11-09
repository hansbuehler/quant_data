"""Quote and expiry diagnostics for saved JAX affine DYSANOS fits."""

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
    for name in ("completion", "panel", "archive", "initialization", "fits", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--model-business-dtes", type=int, nargs="+")
    parser.add_argument("--allow-zero-backbone", action="store_true")
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError("A managed GPU allocation is required")
    geometry = _geometry(_geometry_from_completion(json.loads(args.completion.read_text()), requested_model_business_dtes=None), lower=-4, upper=3)
    geometry = with_model_business_dtes(geometry, args.model_business_dtes, allow_zero_backbone=args.allow_zero_backbone)
    with np.load(args.initialization / "states.npz") as saved:
        if "model_business_dtes" in saved and not np.array_equal(saved["model_business_dtes"], geometry.model_business_dtes):
            raise ValueError("Initialization and model expiry grids differ")
        dates = tuple(date.fromisoformat(str(x)) for x in saved["dates"])
    quote = load_quotes(dates, args.panel, args.archive, geometry)
    times = geometry.model_expiries_y * 252
    upper, lower, alpha, before = (quote[name] for name in ("upper", "lower", "alpha", "before"))
    dte = np.rint((1-alpha)*np.where(before, 0., times[lower]) + alpha*times[upper]).astype(np.int32)
    tables = []
    for mode in ("variance", "volatility"):
        with np.load(args.fits / f"{mode}-final.npz") as saved:
            params = {name: jnp.asarray(saved[name]) for name in ("b", "B", "h")}
        _, predict = make_functions(geometry, mode)
        chunks = []
        for offset in range(0, len(dates), 5):
            idx = np.arange(offset, min(offset+5, len(dates)))
            block = {name: jnp.asarray(value[idx]) for name, value in quote.items()}
            raw = params["b"][None, :] + params["h"][idx] @ params["B"]
            price = np.asarray(predict(raw, block)[0])
            mask = quote["mask"][idx]
            bid, ask = quote["bids"][idx], quote["asks"][idx]
            excess = (np.maximum(bid-price, 0) + np.maximum(price-ask, 0)) / np.maximum((ask-bid)/2, 1e-12)
            chunks.append(pd.DataFrame({"dte": dte[idx][mask], "excess": excess[mask]}))
            if (offset+5) % 50 == 0:
                print(f"mode={mode} days={offset+5}/{len(dates)}", flush=True)
        frame = pd.concat(chunks, ignore_index=True)
        rows = []
        for expiry, group in [(0, frame), *frame.groupby("dte", sort=True)]:
            outside = group.excess[group.excess > 0]
            rows.append({"interpolation": mode, "dte": int(expiry), "quotes": len(group),
                         "inside_percent": 100*(1-len(outside)/len(group)),
                         "conditional_mean_excess_half_spreads": float(outside.mean()) if len(outside) else 0.,
                         "conditional_median_excess_half_spreads": float(outside.median()) if len(outside) else 0.})
        tables.extend(rows)
    pd.DataFrame(tables).to_csv(args.output, index=False)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
