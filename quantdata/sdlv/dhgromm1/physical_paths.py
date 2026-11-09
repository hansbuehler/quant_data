"""Prepare matched one-step P paths and causal marks once, on CPU."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from filelock import FileLock

from quantdata.sdlv.daily.front_spot.jax_model import black_mixture_call_prices
from quantdata.sdlv.daily.wide_audit import AuditMonitor, resource_health, sha256
from quantdata.sdlv.dhgromm1.contracts import RETURN_CONVENTION, SCENARIO_SCHEMA
from quantdata.sdlv.dhgromm1.feedback import contract_expiries
from quantdata.sdlv.dhgromm1.observable import market_geometry
from quantdata.sdlv.dhgromm1.persistence import atomic_json
from quantdata.sdlv.dhgromm1.rolling_simulation import generate_chunk


def load_risk_price(*, fit: Path, calibration: Path) -> tuple[np.ndarray, dict]:
    """Bind an existing strictly-prior physical fit to the frozen diffusion."""
    report = json.loads(fit.with_suffix(".json").read_text())
    if (report["training_end"] >= "2025-08-29" or not report["diffusion_matches_frozen"]
            or report["convention"] != "dW_Q = dW_P + lambda dt; constant density-weighted projection of historical AR drift"):
        raise ValueError("physical fit has incompatible timing or measure convention")
    with np.load(fit,allow_pickle=False) as physical, np.load(calibration/"diffusion.npz",allow_pickle=False) as q:
        for key in ("origin_indices","endpoint_indices"):
            np.testing.assert_array_equal(physical[key],q[key])
        risk_price = physical["constant_market_price_of_risk"].copy()
    if risk_price.shape != (8,) or not np.isfinite(risk_price).all():
        raise ValueError("expected eight finite historical Brownian risk prices")
    np.testing.assert_array_equal(risk_price,np.asarray(report["constant_market_price_of_risk"]))
    return risk_price, report


@jax.jit
def price_existing(support, tower, background, strikes, remaining):
    density = tower[:,remaining-1,:]
    return black_mixture_call_prices(support,density,strikes[...,None],background,remaining)[...,0]


class MarketObserver:
    """Only observed <=t cash marks leave this callback; no replay or latent cache."""

    def __init__(self, *, paths: int, batch_size: int, levels: tuple[float,...]) -> None:
        self.offsets = market_geometry()[0]
        self.values = np.empty((paths,int(self.offsets[-1])),dtype=np.float32)
        self.expiry = contract_expiries()
        tenors = np.stack((np.ones(20),np.full(20,5),np.minimum(10,25-np.arange(20))),-1)
        ratios = np.exp(.20*np.sqrt(tenors[:,:,None]/252)*np.array(levels))
        self.ratios = np.c_[np.zeros(20),np.ones(20),ratios.reshape(20,15),np.zeros((20,4))]
        self.spots = np.zeros((batch_size,20))
        self.strikes = np.zeros((batch_size,20,21))
        self.openings = np.zeros_like(self.strikes)

    def opening(self, day, start, valid, opening) -> None:
        self.openings[:,day] = opening

    def __call__(self, day, start, valid, support, tower, background, spot) -> None:
        current = np.asarray(spot)
        if day == 0:
            self.spots.fill(0)
            self.strikes.fill(0)
            self.openings.fill(0)
        self.spots[:,day] = current
        if day:
            expiry = self.expiry[:day]
            marks = np.zeros((len(spot),day,21))
            marks[:,:,0] = current[:,None]
            cohort,instrument = np.nonzero(expiry[:,1:17]>day)
            instrument += 1
            remaining = expiry[cohort,instrument]-day
            marks[:,cohort,instrument] = np.asarray(price_existing(support,tower,background,
                jnp.asarray(self.strikes[:,cohort,instrument]),jnp.asarray(remaining)))
            cohort,instrument = np.nonzero(expiry[:,1:17]<=day)
            instrument += 1
            marks[:,cohort,instrument] = np.maximum(
                self.spots[:,expiry[cohort,instrument]]-self.strikes[:,cohort,instrument],0)
            for instrument in range(17,21):
                dates = expiry[:,instrument]
                forward = np.asarray(background)[:,None]*np.einsum("bci,i->bc",
                    np.asarray(tower)[:,np.maximum(dates-day,1)-1],np.asarray(support))
                settled = self.spots[:,np.minimum(dates,day)]
                marks[:,:,instrument] = np.where((dates>day)[None],forward,settled)
            changes = marks-self.openings[:,:day]
            if not np.isfinite(changes).all():
                raise FloatingPointError(f"nonfinite observable marks at day {day}")
            self.values[start:start+valid,self.offsets[day]:self.offsets[day+1]] = changes[:valid].reshape(valid,-1)
        self.strikes[:,day] = current[:,None]*self.ratios[day]


def prepare(*, reference: Path, fit: Path, output: Path) -> dict:
    """Two 100k paired samples; retain causal train marks in the same simulation."""
    original = json.loads((reference/"identity.json").read_text())
    if original["substeps"] != 1 or original["instruments"] != 21 or original["paths_per_sample"] != 100_000:
        raise ValueError("P comparison requires the same 100k one-step, 21-instrument geometry")
    calibration = Path(original["calibration"])
    if sha256(calibration/"diffusion.npz") != original["calibration_sha256"]:
        raise ValueError("frozen calibration changed")
    risk_price, physical = load_risk_price(fit=fit,calibration=calibration)
    scenarios,cache = output/"scenarios",output/"market-cache"
    scenarios.mkdir(parents=True,exist_ok=True)
    cache.mkdir(exist_ok=True)
    identity = dict(original,schema=SCENARIO_SCHEMA,return_convention=RETURN_CONVENTION,measure="P",
        source_sha256=sha256(Path(__file__).with_name("rolling_simulation.py")),
        physical_fit=str(fit.resolve()),physical_fit_sha256=sha256(fit),physical_fit_report=physical,
        return_definition="(closing_cash_mark-opening_cash_mark)/S_open; expiry or day20 MTM",
        gain_definition="sum(floating quantities * floating gains); do NOT subtract P means")
    identity_path = scenarios/"identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("P scenario identity changed; choose a new run directory")
    atomic_json(identity_path,identity)
    cache_identity = dict(schema="dhgromm1-causal-market-marks-v1",paths=100_000,sample="train",
        scenarios=str(scenarios.resolve()),scenario_identity_sha256=sha256(identity_path),
        calibration_sha256=identity["calibration_sha256"],generator_sha256=identity["source_sha256"],
        cache_code_sha256=sha256(Path(__file__)),values_per_path=int(market_geometry()[0][-1]),dtype="float32",
        meaning="current mark minus opening mark; frozen at observed settlement; generated alongside P paths",
        no_simulator_state_saved=True)
    cache_manifest = cache/"identity.json"
    if cache_manifest.exists() and json.loads(cache_manifest.read_text()) != cache_identity:
        raise ValueError("P cache identity changed; choose a new run directory")
    atomic_json(cache_manifest,cache_identity)
    monitor = AuditMonitor(output)
    monitor.thread.start()
    records = []
    started = time.monotonic()
    try:
        with FileLock(output/"prepare.lock",timeout=0):
            completed = 0
            for sample_index,sample in enumerate(("train","evaluation")):
                folder = scenarios/sample
                folder.mkdir(exist_ok=True)
                for start in range(0,100_000,original["chunk_size"]):
                    count = min(original["chunk_size"],100_000-start)
                    path = folder/f"{start:06d}.npz"
                    mark_path = cache/path.name
                    monitor.stage(f"prepare-P-{sample}",completed,200_000)
                    if not path.exists() or (sample=="train" and not mark_path.exists()):
                        observer = MarketObserver(paths=count,batch_size=original["batch_size"],levels=tuple(original["z"])) if sample=="train" else None
                        prices,returns,payoff = generate_chunk(calibration=calibration,paths=count,
                            batch_size=original["batch_size"],substeps=1,
                            seed=np.random.SeedSequence([original["seed"],sample_index,start]),
                            strike_levels=tuple(original["z"]),include_forwards=True,market_price_of_risk=risk_price,
                            market_sink=observer,opening_sink=observer.opening if observer else None)
                        np.testing.assert_allclose(payoff,np.abs(prices[:,20,0]/prices[:,10,0]-1),atol=0,rtol=0)
                        error = float(np.max(np.abs(2*returns[:,10,1]-returns[:,10,17]-
                            (payoff-(1+(2*prices[:,10,1]-prices[:,10,17])/prices[:,10,0])))))
                        if error>1e-10:
                            raise ValueError("P floating straddle replication failed")
                        temporary = path.with_suffix(".tmp.npz")
                        np.savez_compressed(temporary,prices=prices,returns=returns,payoff=payoff)
                        temporary.replace(path)
                        if observer is not None:
                            record = dict(start=start,count=count,scenario_chunk_sha256=sha256(path),
                                replay_max_errors=[0.,0.,0.],replay_backend="cpu; no replay; jointly generated",straddle_error=error)
                            temporary = mark_path.with_suffix(".tmp.npz")
                            np.savez_compressed(temporary,changes=observer.values,record=json.dumps(record))
                            temporary.replace(mark_path)
                    if sample=="train":
                        with np.load(mark_path,allow_pickle=False) as saved:
                            record = json.loads(str(saved["record"].item()))
                            if saved["changes"].shape!=(count,3990) or record["scenario_chunk_sha256"]!=sha256(path):
                                raise ValueError("P cache chunk identity/geometry differs")
                        records.append(record)
                    completed += count
                    elapsed = time.monotonic()-started
                    print(json.dumps(dict(stage="P-scenario-progress",completed=completed,total=200_000,
                        elapsed_seconds=elapsed,estimated_remaining_seconds=elapsed*(200_000-completed)/completed)),flush=True)
                if sample=="train":
                    atomic_json(cache/"complete.json",dict(status="complete",**cache_identity,chunks=records))
            report = dict(status="complete",measure="P",scenarios=str(scenarios),cache=str(cache),paths_per_sample=100_000)
            atomic_json(output/"status.json",report)
            return report
    finally:
        monitor.done.set()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--fit",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    root = Path(os.environ["QUANTDATA_ROOT"]).resolve()
    if not args.output.resolve().is_relative_to(root) or args.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("P preparation must use a unique external project directory")
    checks = resource_health()
    if not checks or any(c["health"]!="OK" for c in checks):
        raise RuntimeError("resource health must be OK before starting material P preparation")
    print(json.dumps(prepare(reference=args.reference,fit=args.fit,output=args.output)),flush=True)


if __name__ == "__main__":
    main()
