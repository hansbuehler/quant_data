"""Compare fitted DYSANOS prices with SANOS on the fitted model's own grid."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, ndtr

from quantdata.dysanos.flex.run_all_history_joint_direct import _geometry_from_completion
from quantdata.dysanos.flex.sanos import make_supports_from_total_vols
from quantdata.dysanos.v11.jax_geometry import with_model_business_dtes
from quantdata.dysanos.v11.licensed_market import load_certified_pricer
from quantdata.dysanos.v11.run_rolling_direct_ar import _geometry
from quantdata.sanos.tridiag import solve_sanos_tridiagonal


def grid_prices(raw: np.ndarray, geometry) -> tuple[np.ndarray, np.ndarray]:
    """Return shared support and normalized calls at each model expiry."""
    n, t = geometry.num_inner_strikes, geometry.num_model_expiries
    blocks = raw[:t * (n + 1)].reshape(t, n + 1)
    variance = geometry.min_dlvsq + np.exp(np.minimum(blocks[:, :n], geometry.maximum_log_dlvsq))
    forward = geometry.min_forward_vol + (geometry.max_forward_vol - geometry.min_forward_vol) * expit(blocks[:, n])
    steps = np.diff(geometry.model_expiries_y, prepend=0.)
    total = np.cumsum(forward * np.sqrt(steps))
    eta = geometry.min_eta + (geometry.max_eta - geometry.min_eta) * expit(raw[-2:])
    backbone = np.cumsum(forward * np.r_[eta[0], np.full(t - 1, eta[1])] * np.sqrt(steps))
    support = make_supports_from_total_vols(total, geometry=geometry)[0]
    inner = support[1:-1]
    law = np.zeros(n + 2)
    law[np.argmin(abs(support - 1.))] = 1.
    result = []
    for i, step in enumerate(steps):
        weight = variance[i] * inner**2 * step / (support[2:] - support[:-2])
        law = solve_sanos_tridiagonal(w_minus=weight / (inner - support[:-2]), w_plus=weight / (support[2:] - inner), rhs=law)
        w = max(backbone[i], 1e-10)
        z = (np.log(support[None, :] / inner[:, None]) + .5 * w * w) / w
        calls = support[None, :] * ndtr(z) - inner[:, None] * ndtr(z - w)
        result.append(calls @ law)
    return inner, np.asarray(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("completion", "archive", "initialization", "checkpoint", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--model-business-dtes", type=int, nargs="+", required=True)
    parser.add_argument("--last-dates", type=int)
    args = parser.parse_args()
    geometry = _geometry(_geometry_from_completion(json.loads(args.completion.read_text()), requested_model_business_dtes=None), lower=-4, upper=3)
    geometry = with_model_business_dtes(geometry, args.model_business_dtes, allow_zero_backbone=True)
    with np.load(args.initialization / "states.npz") as saved:
        dates = [date.fromisoformat(str(day)) for day in saved["dates"]]
        if not np.array_equal(saved["model_business_dtes"], geometry.model_business_dtes):
            raise ValueError("Initialization and model expiry grids differ")
    with np.load(args.checkpoint) as saved:
        raw = saved["states"] if "states" in saved else saved["b"] + saved["h"] @ saved["B"]
        iteration = int(saved["iteration"])
    if len(raw) != len(dates):
        raise ValueError("Checkpoint states do not match initialization dates")
    if args.last_dates is not None:
        if not 0 < args.last_dates <= len(dates):
            raise ValueError("last-dates must be within the available dates")
        dates, raw = dates[-args.last_dates:], raw[-args.last_dates:]
    errors = []
    for index, (day, state) in enumerate(zip(dates, raw, strict=True), 1):
        source, _ = load_certified_pricer(day, archive=args.archive)
        strikes, fit = grid_prices(state, geometry)
        reference = source.price(geometry.model_expiries_y[:, None], strikes[None, :], mode="surface")
        errors.append(fit - reference)
        if index % 25 == 0:
            print(f"dates={index}/{len(dates)}", flush=True)
    error = np.asarray(errors)
    rows = []
    for index, dte in [(None, 0), *enumerate(args.model_business_dtes)]:
        values = np.abs(error if index is None else error[:, index, :]).ravel()
        rows.append({"dte": dte, "n": len(values), "mean_abs_normalized_price_error": float(values.mean()),
                     "median_abs_normalized_price_error": float(np.median(values)), "p95_abs_normalized_price_error": float(np.quantile(values, .95)),
                     "max_abs_normalized_price_error": float(values.max()), "iteration": iteration})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
