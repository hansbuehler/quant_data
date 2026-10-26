"""CPU-only interim VIX from an immutable training-state snapshot.

Reuse an audited market constituent strip, including its observed bid cutoffs,
rates and expiry minutes. Never modify the running training checkpoint or
pretend an interim checkpoint is a completed training report.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.sdlv.daily.front_spot.estimation import BACKGROUND_ANNUAL_VOL, decode_dlvsq
from quantdata.sdlv.daily.front_spot.jax_model import decode_density_tower
from quantdata.sdlv.daily.plot_vix import run as plot_comparison
from quantdata.sdlv.daily.vix_comparison import (
    _metrics,
    _sha256,
    interpolate_density,
    log_contract_total_variance,
    mixture_prices,
)
from quantdata.vixdata.formula import VixStrip, constant_maturity_vix, discrete_total_variance


def reprice_saved_strip(
    term: dict, contracts: pl.DataFrame, support: np.ndarray, tower: np.ndarray,
) -> tuple[float | None, float | None]:
    """Price the exact saved market strip; refuse inconsistent quadrature weights."""
    contracts = contracts.sort("strike")
    strikes = contracts["strike"].to_numpy()
    deltas = contracts["delta_strike"].to_numpy()
    if len(strikes) < 3 or len(strikes) != term["selected_strikes"] or not np.all(np.diff(strikes) > 0):
        raise ValueError("Saved strip has missing or unordered strikes")
    expected = np.r_[strikes[1] - strikes[0], (strikes[2:] - strikes[:-2]) / 2, strikes[-1] - strikes[-2]]
    if not np.allclose(deltas, expected, rtol=0, atol=1e-10):
        raise ValueError("Saved strip deltaK does not match its selected strikes")
    strip = VixStrip(indices=np.arange(len(strikes)), strikes=strikes, delta_strikes=deltas,
                     prices=contracts["market_price"].to_numpy(), forward=term["forward"], k0=term["k0"],
                     put_zero_bid_stop=term["put_zero_bid_stop"], call_zero_bid_stop=term["call_zero_bid_stop"])
    market = discrete_total_variance(strip, discount_factor=term["discount_factor"])
    if not np.isclose(market, term["market_total_variance"], rtol=0, atol=1e-12):
        raise ValueError("Saved market strip does not reproduce its audited total variance")
    business_days = term["business_days"]
    if not 1 <= business_days <= len(tower):
        return None, None
    density = interpolate_density(tower, business_days)
    calls, puts = mixture_prices(support, density, strikes / strip.forward, business_days=business_days)
    prices = np.where(strikes < strip.k0, puts, calls)
    at_k0 = strikes == strip.k0
    prices[at_k0] = (calls[at_k0] + puts[at_k0]) / 2
    prices *= strip.forward * term["discount_factor"]
    discrete = discrete_total_variance(strip, discount_factor=term["discount_factor"], prices=prices)
    analytic = log_contract_total_variance(support, density, business_time_years=business_days / DAYS_PER_YEAR)
    return discrete, analytic


def run(*, checkpoint_state: Path, dataset: Path, baseline_directory: Path,
        filtered_directory: Path, output_directory: Path) -> dict:
    """Evaluate a frozen interim state without interrupting or changing training."""
    if jax.default_backend() != "cpu":
        raise RuntimeError("Use JAX_PLATFORMS=cpu; training retains its GPU allocation")
    started = time.monotonic()
    with np.load(checkpoint_state, allow_pickle=False) as saved:
        identity = json.loads(str(saved["identity"].item()))
        epoch = int(saved["epoch"])
        raw = np.asarray(saved["raw"], dtype=np.float64)
    if identity["dataset_sha256"] != _sha256(dataset):
        raise ValueError("Snapshot does not belong to the specified dataset")
    with np.load(dataset, allow_pickle=False) as data:
        supports = np.asarray(data["supports"], dtype=np.float64)
        dates = data["dates"].astype(str).tolist()
        np.testing.assert_allclose(data["model_times_y"], np.arange(1, 26) / DAYS_PER_YEAR, rtol=1e-7, atol=0)
        if not np.isclose(float(data["backbone_annual_vol"]), BACKGROUND_ANNUAL_VOL):
            raise ValueError("Unexpected smoothing volatility")
    if raw.shape != (len(dates), 25, supports.shape[-1] - 2) or not np.all(np.isfinite(raw)):
        raise ValueError("Invalid snapshot geometry or parameters")
    output_directory.mkdir(parents=True, exist_ok=False)
    with jax.enable_x64():
        decode = jax.jit(jax.vmap(lambda s, v: decode_density_tower(s, v, dt=1 / DAYS_PER_YEAR)))
        towers = np.asarray(decode(jnp.asarray(supports), jnp.asarray(decode_dlvsq(raw))))
    mass_error = float(np.max(np.abs(towers.sum(axis=-1) - 1)))
    forward_error = float(np.max(np.abs(np.einsum("dti,di->dt", towers, supports) - 1)))
    if towers.dtype != np.float64 or not np.all(np.isfinite(towers)) or towers.min() < 0 or max(mass_error, forward_error) > 1e-10:
        raise ValueError("Snapshot fails strict float64 density validation")
    base = pl.read_parquet(baseline_directory / "vix_comparison.parquet")
    audits = pl.read_parquet(baseline_directory / "expiry_audit.parquet")
    contracts = pl.read_parquet(baseline_directory / "selected_contracts.parquet").partition_by(
        ["date", "expiry", "is_am"], as_dict=True)
    day_terms = {key[0]: group.sort("calendar_days").to_dicts()
                 for key, group in audits.partition_by("date", as_dict=True).items()}
    if base["date"].cast(pl.String).to_list() != dates:
        raise ValueError("Baseline date order does not match the training dataset")
    records, terms = [], []
    for index, (day, support, tower) in enumerate(zip(dates, supports, towers, strict=True)):
        record = dict(date=date.fromisoformat(day), gromm_discrete=None, gromm_friday_continuous=None)
        pair = []
        for term in day_terms.get(day, []):
            discrete, analytic = reprice_saved_strip(term, contracts[(day, term["expiry"], term["is_am"])], support, tower)
            pair.append((term, discrete, analytic))
            terms.append(dict(date=day, expiry=term["expiry"], is_am=term["is_am"],
                              calendar_days=term["calendar_days"], business_days=term["business_days"],
                              gromm_total_variance=discrete, gromm_full_strike_total_variance=analytic))
        if len(pair) == 2 and all(entry[1] is not None for entry in pair):
            near, far = pair
            for key, position in (("gromm_discrete", 1), ("gromm_friday_continuous", 2)):
                record[key] = constant_maturity_vix(near_days=near[0]["calendar_days"], next_days=far[0]["calendar_days"],
                                                    near_total_variance=near[position], next_total_variance=far[position])
        records.append(record)
        if index == 0 or (index + 1) % 100 == 0 or index + 1 == len(dates):
            elapsed = time.monotonic() - started
            print(json.dumps(dict(epoch=epoch, completed=index + 1, total=len(dates), elapsed_seconds=elapsed,
                                  estimated_remaining_seconds=elapsed * (len(dates) - index - 1) / (index + 1))), flush=True)
    frame = base.select("date", "cboe_vix", "market_discrete").join(pl.from_dicts(records, infer_schema_length=None), on="date")
    total_epochs = identity["train_config"]["initial_epochs"] + identity["train_config"]["continuation_epochs"]
    report = dict(status="interim", model_label=f"{identity['model_label']} — INTERIM epoch {epoch}/{total_epochs}",
                  epoch=epoch, total_epochs=total_epochs, training_identity=identity,
                  checkpoint_state=str(checkpoint_state), checkpoint_sha256=_sha256(checkpoint_state),
                  dataset=str(dataset), dataset_sha256=identity["dataset_sha256"],
                  code_sha256={str(Path(__file__)): _sha256(Path(__file__))},
                  baseline_hashes={name: _sha256(baseline_directory / name) for name in
                                   ("vix_comparison.parquet", "expiry_audit.parquet", "selected_contracts.parquet")},
                  validation=dict(minimum_probability=float(towers.min()), maximum_mass_error=mass_error, maximum_forward_error=forward_error),
                  metrics={key: _metrics(frame[key].to_numpy(), frame["cboe_vix"].to_numpy())
                           for key in ("gromm_discrete", "gromm_friday_continuous")},
                  elapsed_seconds=time.monotonic() - started,
                  conventions=["Frozen intermediate checkpoint; training continues separately and these are not final fit results.",
                               "Reprices the verified original market-selected strip, preserving bid cutoffs, F/K0, deltaK, rates and expiry interpolation.",
                               "No extrapolation beyond the 25-business-day model horizon."])
    frame.write_parquet(output_directory / "vix_comparison.parquet")
    frame.write_csv(output_directory / "vix_comparison.csv")
    pl.from_dicts(terms, infer_schema_length=None).write_parquet(output_directory / "expiry_audit.parquet")
    (output_directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    plot_comparison(model_directory=output_directory, filtered_directory=filtered_directory,
                    output_directory=output_directory / "four-series")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint-state", "dataset", "baseline-directory", "filtered-directory", "output-directory"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
