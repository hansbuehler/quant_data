"""Shared-latent, active-maturity Q samples for two real-contract hedge books."""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import threading
import time
from dataclasses import asdict
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from filelock import FileLock
from scipy.linalg import qr

from quantdata.gromm2.bridge import front_density_volatility
from quantdata.gromm2.dlv import prepare_sparse_transports, risk_neutral_log_drift
from quantdata.gromm2.dynamics import interpolate_loading_expiries
from quantdata.gromm2.forward_hedging import ForwardScenarioConfig, independent_normal_block
from quantdata.gromm2.legacy_forward_model import load_legacy_ar1_model
from quantdata.gromm2.market_books import (
    basket_contracts,
    contract_marks,
    factor_baskets,
    primary_contracts,
    vanilla_kernel,
)
from quantdata.gromm2.rolling_audit import check_resource_headroom
from quantdata.gromm2.rolling_calibration import write_json
from quantdata.gromm2.scenario_statistics import PairMoments
from quantdata.gromm2.simulation import (
    SimulationState,
    localize_loadings_by_density,
    one_day_risk_neutral_step,
)

CASES = (("black5", 0.05), ("black0", 0.0))


def inception_baskets(arrays: dict) -> tuple[list[np.ndarray], list[dict], dict]:
    """Price actual option diffusion exposures under each background kernel."""
    densities, supports = map(jnp.asarray, (arrays["densities"], arrays["supports"]))
    law = {name: jnp.asarray(value) for name, value in arrays["ar1_parameters"].items()}
    scales = law["fixed_factor_scales"] * law["scale_calibration"] * jnp.exp(jnp.clip(
        0.5 * (law["variance_intercept"] + law["variance_slope"] * law["terminal_log_squared"]),
        -law["log_scale_bound"], law["log_scale_bound"]))
    loading = localize_loadings_by_density(jnp.asarray(arrays["loadings"]) * scales,
        densities[1:], relative_floor=1e-4)
    front = jnp.zeros((supports.shape[-1], loading.shape[-1])).at[:, 0].set(
        front_density_volatility(densities[0], remaining_time=jnp.asarray(1 / 252)))
    _, future_loading, _ = risk_neutral_log_drift(supports, prepare_sparse_transports(arrays["transports"]),
        densities[0], jnp.asarray(arrays["dlvsq"][1:]), jnp.asarray(np.diff(arrays["expiries_y"])),
        loading, front, linear_solver="pcr")
    all_loading = jnp.concatenate((front[None], future_loading))
    dtes = np.rint(arrays["expiries_y"] * 252).astype(int)
    book = basket_contracts(dtes)
    indices = jnp.asarray(np.searchsorted(dtes, book["expiries"]))
    latent = supports[indices]
    weights, reports, exposures, securities = [], [], [], []
    for _, volatility in CASES:
        kernel, delta = vanilla_kernel(latent, jnp.asarray(book["strikes"])[:, None],
            volatility * jnp.sqrt(jnp.asarray(book["expiries"])[:, None] / 252), jnp.asarray(book["calls"])[:, None])
        diffusion = jnp.einsum("cn,cnf->cf", kernel, all_loading[indices])
        background = volatility * jnp.sum(delta * latent * densities[indices], axis=-1)
        exposure = np.asarray(jnp.column_stack((diffusion, background)))
        exposures.append(exposure)
        weight, report = factor_baskets(exposure, residual_factors=loading.shape[-1] - 1)
        call_kernel, call_delta = vanilla_kernel(supports[0], jnp.asarray(1.0),
            jnp.asarray(volatility / np.sqrt(252)), jnp.asarray(True))
        call_exposure = np.r_[np.asarray(call_kernel @ front),
            float(volatility * jnp.sum(call_delta * supports[0] * densities[0]))]
        spot_exposure = np.r_[np.asarray(supports[0] @ front), volatility]
        securities.append((spot_exposure, call_exposure))
        combined = np.vstack((weight.T @ exposure, spot_exposure, call_exposure))
        singular = np.linalg.svd(combined, compute_uv=False)
        report.update(combined_initial_singular_values=singular.tolist(),
            combined_initial_rank=int(np.linalg.matrix_rank(combined, tol=1e-9 * singular[0])),
            expected_drivers=loading.shape[-1] + int(volatility > 0))
        weights.append(weight)
        reports.append(report | {"candidate_book": {name: value.tolist() for name, value in book.items()}})
    selected = []
    for exposure, report in zip(exposures, reports, strict=True):
        normalized = exposure / np.maximum(np.linalg.norm(exposure, axis=0), 1e-14)
        _, _, pivot = qr(normalized.T, pivoting=True, mode="economic")
        selected.extend(pivot[:2 * report["expected_drivers"]])
    selected = np.unique(selected)
    selected_book = {name: value[selected] for name, value in book.items()}
    for index, exposure in enumerate(exposures):
        weight, report = factor_baskets(exposure[selected], residual_factors=loading.shape[-1] - 1)
        combined = np.vstack((weight.T @ exposure[selected], *securities[index]))
        singular = np.linalg.svd(combined, compute_uv=False)
        report.update(combined_initial_singular_values=singular.tolist(),
            combined_initial_rank=int(np.linalg.matrix_rank(combined, tol=1e-9 * singular[0])),
            expected_drivers=loading.shape[-1] + int(CASES[index][1] > 0),
            full_candidate_count=len(book["strikes"]), selected_candidate_count=len(selected),
            selection="Union of inception column-normalized pivoted-QR contract subsets for both backgrounds; no future returns.",
            candidate_book={name: value.tolist() for name, value in selected_book.items()})
        weights[index], reports[index] = weight, report
    return weights, reports, selected_book


class PairedSimulator:
    """Compile each strictly active daily tower once and reuse it across batches."""

    def __init__(self, arrays: dict, config: ForwardScenarioConfig):
        self.arrays, self.config = arrays, config
        self.dtes = np.rint(arrays["expiries_y"] * 252).astype(int)
        if not np.array_equal(self.dtes[:21], np.arange(1, 22)):
            raise ValueError("daily model expiries through 21 business days are required")
        self.primary = primary_contracts()
        self.weights, self.basket_reports, self.candidates = inception_baskets(arrays)
        basis, _ = np.linalg.qr(arrays["loadings"].reshape(-1, arrays["loadings"].shape[-1]))
        self.basis = jnp.asarray(np.concatenate((np.zeros((1, arrays["dlvsq"].shape[-1], basis.shape[-1])),
            basis.reshape(arrays["loadings"].shape))))
        self.law = {name: jnp.asarray(value) for name, value in arrays["ar1_parameters"].items()}
        self.kernels = [self._day_kernel(day) for day in range(20)]
        self.quote_kernels = [self._quote_kernel(day) for day in range(21)]

    def _day_kernel(self, day: int):
        arrays, law, config = self.arrays, self.law, self.config
        supports = jnp.asarray(arrays["supports"][day:])
        transport = prepare_sparse_transports(arrays["transports"][day:])
        intervals = jnp.asarray(np.diff(arrays["expiries_y"])[day:])
        loading = jnp.asarray(interpolate_loading_expiries(arrays["loadings"],
            arrays["loading_reference_days"], self.dtes[day + 1:] - day))
        reference, basis = jnp.log(jnp.asarray(arrays["dlvsq"][day:])), self.basis[day:].at[0].set(0.0)

        @jax.jit
        def advance(front, dlvsq, ar_state, cumulative, normals):
            predicted = law["variance_intercept"] + law["variance_slope"] * ar_state
            scales = law["fixed_factor_scales"] * law["scale_calibration"] * jnp.exp(
                jnp.clip(0.5 * predicted, -law["log_scale_bound"], law["log_scale_bound"]))
            logarithm = jnp.log(dlvsq)
            coordinates = jnp.einsum("bjn,jnf->bf", logarithm - reference, basis)
            means = jnp.pad(jnp.mean(logarithm, axis=-1), ((0, 0), (0, day)))
            features = jnp.concatenate((front, coordinates, means, cumulative, ar_state, scales,
                jnp.full((len(front), 1), day / 20), jnp.full((len(front), 1), day >= 10)), axis=-1)
            result = one_day_risk_neutral_step(supports, transport, intervals,
                SimulationState(front, dlvsq, jnp.ones(len(front))), loading, normals, jnp.zeros(len(front)),
                remaining_front_days=jnp.asarray(1), substeps=config.substeps,
                background_annual_volatility=0.0, linear_solver=config.linear_solver,
                density_loading_floor=config.density_loading_floor, factor_scales=scales)
            next_ar = jnp.log((scales * result.daily_factor_score / law["fixed_factor_scales"])**2
                + law["squared_score_floor"])
            diagnostic = jnp.stack((result.drift_taming_fraction,
                jnp.max(jnp.abs(jnp.sum(result.density_tower, axis=-1) - 1)),
                jnp.max(jnp.abs(jnp.sum(result.density_tower * supports, axis=-1) - result.latent_spot[:, None])),
                jnp.min(result.density_tower)))
            return (result.state.front_density, result.state.dlvsq[:, 1:], next_ar,
                cumulative + normals.sum(axis=0) / jnp.sqrt(252 * config.substeps),
                result.density_tower, result.latent_spot, features.astype(jnp.float32), diagnostic)

        return advance

    def _quote_kernel(self, day: int):
        offset = max(day - 1, 0)
        supports = jnp.asarray(self.arrays["supports"][offset:])
        primary, candidates, dtes = self.primary, self.candidates, self.dtes
        row = day - offset

        @jax.jit
        def quote(tower, backgrounds, spots, birth_spots):
            outputs = []
            for case_index, (_, volatility) in enumerate(CASES):
                spot, background = spots[case_index], backgrounds[case_index]
                strikes = birth_spots[case_index][:, jnp.asarray(primary["starts"])] * jnp.asarray(primary["levels"])
                primary_marks = contract_marks(supports=supports, densities=tower, background=background,
                    spot=spot, strikes=strikes, expiries=primary["expiries"], calls=primary["calls"],
                    model_dtes=dtes, offset=offset, day=day, volatility=volatility)
                primary_marks = jnp.where(jnp.asarray(primary["starts"])[None] <= day, primary_marks, 0.0)
                candidate_strikes = jnp.broadcast_to(jnp.asarray(candidates["strikes"]), (len(spot), len(candidates["strikes"])))
                marks = contract_marks(supports=supports, densities=tower, background=background,
                    spot=spot, strikes=candidate_strikes, expiries=candidates["expiries"], calls=candidates["calls"],
                    model_dtes=dtes, offset=offset, day=day, volatility=volatility)
                portfolio_marks = marks @ jnp.asarray(self.weights[case_index])
                latent = background[:, None] * supports[row]
                kernel, _ = vanilla_kernel(latent, spot[:, None], jnp.asarray(volatility / np.sqrt(252)), jnp.asarray(True))
                daily_call = jnp.sum(kernel * tower[:, row], axis=-1)
                outputs.append((primary_marks, portfolio_marks, daily_call))
            return tuple(jnp.stack([record[index] for record in outputs]) for index in range(3))

        return quote

    def simulate(self, factors: np.ndarray, background_normals: np.ndarray, progress: dict) -> dict:
        count = factors.shape[2]
        arrays = self.arrays
        front = jnp.broadcast_to(jnp.asarray(arrays["densities"][0]), (count, arrays["supports"].shape[-1]))
        dlvsq = jnp.broadcast_to(jnp.asarray(arrays["dlvsq"]), (count,) + arrays["dlvsq"].shape)
        ar_state = jnp.broadcast_to(self.law["terminal_log_squared"], (count, len(self.law["terminal_log_squared"])))
        cumulative = jnp.zeros_like(ar_state)
        log_background = np.concatenate((np.zeros((count, 1)), np.cumsum(background_normals.T * 0.05 / np.sqrt(252) - 0.5 * 0.05**2 / 252, axis=1)), axis=1)
        backgrounds = np.stack((np.exp(log_background), np.ones_like(log_background)))
        spots = np.ones((2, count, 21))
        births = np.ones((2, count, 11))
        initial = jnp.broadcast_to(jnp.asarray(arrays["densities"]), (count,) + arrays["densities"].shape)
        prices, portfolio_prices, daily_calls = self.quote_kernels[0](initial, jnp.asarray(backgrounds[:, :, 0]), jnp.asarray(spots[:, :, 0]), jnp.asarray(births))
        price_history, portfolio_history, call_history = [np.asarray(prices)], [np.asarray(portfolio_prices)], [np.asarray(daily_calls)]
        observations, diagnostics = [], []
        for day in range(20):
            progress.update(stage="compiling-or-advancing-active-tower", day=day, total_days=20)
            front, dlvsq, ar_state, cumulative, tower, latent, features, diagnostic = self.kernels[day](
                front, dlvsq, ar_state, cumulative, jnp.asarray(factors[day]))
            spots[:, :, day + 1] = np.asarray(latent)[None] * backgrounds[:, :, day + 1]
            if day + 1 <= 10:
                births[:, :, day + 1] = spots[:, :, day + 1]
            prices, portfolio_prices, daily_calls = self.quote_kernels[day + 1](tower,
                jnp.asarray(backgrounds[:, :, day + 1]), jnp.asarray(spots[:, :, day + 1]), jnp.asarray(births))
            observations.append(np.asarray(features))
            diagnostics.append(np.asarray(diagnostic))
            price_history.append(np.asarray(prices))
            portfolio_history.append(np.asarray(portfolio_prices))
            call_history.append(np.asarray(daily_calls))
        return dict(core_features=np.stack(observations, axis=1), spots=spots, backgrounds=backgrounds,
            option_prices=np.stack(price_history, axis=2), factor_prices=np.stack(portfolio_history, axis=2),
            daily_calls=np.stack(call_history[:20], axis=2), payoff=np.abs(spots[:, :, 20] / spots[:, :, 10] - 1),
            diagnostics=np.stack(diagnostics))


def generate_paired(*, arrays: dict, metadata: dict, directory: Path, split: str, config: ForwardScenarioConfig,
                    simulator: PairedSimulator | None = None) -> Path:
    """Persist one latent sample and both background-priced books without duplication."""
    check_resource_headroom()
    directory.mkdir(parents=True, exist_ok=True)
    source_names = ("paired_forward.py", "market_books.py", "simulation.py", "dlv.py", "bridge.py", "dynamics.py",
        "legacy_forward_model.py", "runtime.py", "tridiagonal.py", "forward_hedging.py", "scenario_statistics.py")
    software = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in source_names}
    software["quantdata/jax/tridiagonal.py"] = hashlib.sha256(
        (Path(__file__).parents[1] / "jax" / "tridiagonal.py").read_bytes()).hexdigest()
    identity = json.dumps(dict(schema="gromm2-paired-market-books-v1", model=metadata, config=asdict(config),
        split=split, software_sha256=software, path_weights="Original equal Q; no empirical measure adjustment"), sort_keys=True)
    target = directory / f"{split}.h5"
    stop, done = threading.Event(), threading.Event()
    progress = dict(stage="initializing", completed=0, total=config.paths)
    started = time.monotonic()
    previous = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGTERM, signal.SIGINT)}

    def heartbeat():
        while not done.wait(60):
            try:
                check_resource_headroom()
            except BaseException:
                stop.set()
            elapsed = time.monotonic() - started
            completed = progress["completed"]
            print(json.dumps(dict(event="paired_forward_progress", split=split, **progress,
                elapsed_seconds=elapsed, remaining_seconds=elapsed * (config.paths - completed) / completed if completed else None)), flush=True)

    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        with FileLock(str(target) + ".lock", timeout=0), h5py.File(target, "a") as stream:
            if "identity" in stream.attrs and stream.attrs["identity"] != identity:
                raise ValueError("paired scenario identity changed; preserve this file and use a new run directory")
            stream.attrs["identity"] = identity
            completed = int(stream.attrs.get("completed_paths", 0))
            progress["completed"] = completed
            if completed == config.paths and (directory / f"{split}_report.json").exists():
                return target
            if simulator is None:
                simulator = PairedSimulator(arrays, config)
            basket_path = directory / "factor_baskets.json"
            if completed and not basket_path.exists():
                raise FileNotFoundError("saved basket identities are required to resume existing path data")
            if basket_path.exists():
                saved_baskets = json.loads(basket_path.read_text())
                saved_weights = [np.asarray(value) for value in saved_baskets["weights"]]
                saved_book = {name: np.asarray(value) for name, value in saved_baskets["candidate_book"].items()}
                if (any(not np.array_equal(first, second) for first, second in zip(saved_weights, simulator.weights, strict=True))
                        or any(not np.array_equal(saved_book[name], simulator.candidates[name]) for name in saved_book)):
                    simulator.weights, simulator.candidates = saved_weights, saved_book
                    simulator.basket_reports = saved_baskets["reports"]
                    simulator.quote_kernels = [simulator._quote_kernel(day) for day in range(21)]
            else:
                write_json(basket_path, dict(reports=simulator.basket_reports,
                    weights=[weight.tolist() for weight in simulator.weights],
                    candidate_book={name: value.tolist() for name, value in simulator.candidates.items()}))
            for name, book in (("primary", simulator.primary), ("candidates", simulator.candidates)):
                for key, value in book.items():
                    dataset = f"{name}/{key}"
                    if dataset not in stream:
                        stream.create_dataset(dataset, data=value)
            moments = {name: PairMoments() for name, _ in CASES}
            returns = {name: PairMoments() for name, _ in CASES}
            primary_returns = {name: PairMoments() for name, _ in CASES}

            def audit_values(group, start, stop):
                spot = np.asarray(group["spots"][start:stop])
                marks = np.asarray(group["factor_prices"][start:stop])
                calls = np.asarray(group["daily_calls"][start:stop])
                daily = np.maximum(np.diff(spot, axis=1), 0.0) - calls
                return np.concatenate((np.diff(marks, axis=1), np.diff(spot, axis=1)[..., None], daily[..., None]), axis=-1) / spot[:, :-1, None]

            for name, _ in CASES:
                for start in range(0, completed, 4096):
                    moments[name].update(stream[f"{name}/payoff"][start:min(start + 4096, completed)])
                    returns[name].update(audit_values(stream[name], start, min(start + 4096, completed)))
                    marks = stream[f"{name}/option_prices"][start:min(start + 4096, completed)]
                    spot = stream[f"{name}/spots"][start:min(start + 4096, completed)]
                    primary_returns[name].update(np.diff(marks, axis=1) / spot[:, :-1, None])

            def snapshot(diagnostic):
                return dict(completed_paths=completed, total_paths=config.paths,
                    prices={name: moment.scalar_summary() for name, moment in moments.items()},
                    factor_book_unconditional={name: moment.screen(np.ones(moment.mean.shape, dtype=bool))
                        for name, moment in returns.items()},
                    primary_book_unconditional={name: moment.screen(
                        (np.arange(20)[:, None] >= simulator.primary["starts"])
                        & (np.arange(20)[:, None] < simulator.primary["expiries"]))
                        for name, moment in primary_returns.items()},
                    maximum_batch_drift_taming=float(np.max(diagnostic[:, 0])), model=metadata)

            while completed < config.paths:
                if stop.is_set():
                    raise InterruptedError("paired sampler stopped at its last complete batch")
                count = min(config.batch_size, config.paths - completed)
                factors, background = independent_normal_block(config=config, split=split, completed=completed,
                    count=count, factors=arrays["loadings"].shape[-1])
                values = simulator.simulate(factors, background, progress)
                if not all(np.all(np.isfinite(value)) for value in values.values()):
                    raise FloatingPointError("non-finite paired scenario; no path discarded")
                diagnostic = values.pop("diagnostics")
                if np.max(diagnostic[:, 1:3]) > 1e-8 or np.min(diagnostic[:, 3]) < -1e-12:
                    raise FloatingPointError("paired density invariants failed")
                items = {"core_features": values.pop("core_features")}
                for case_index, (name, _) in enumerate(CASES):
                    for key, value in values.items():
                        items[f"{name}/{key}"] = value[case_index]
                    moments[name].update(values["payoff"][case_index])
                for name, value in items.items():
                    if name not in stream:
                        stream.create_dataset(name, shape=(config.paths,) + value.shape[1:], dtype=value.dtype)
                    stream[name][completed:completed + count] = value
                for name, _ in CASES:
                    returns[name].update(audit_values(stream[name], completed, completed + count))
                    group = stream[name]
                    primary_returns[name].update(np.diff(group["option_prices"][completed:completed + count], axis=1)
                        / np.asarray(group["spots"][completed:completed + count])[:, :-1, None])
                batch_index = completed // config.batch_size
                if "diagnostics" not in stream:
                    stream.create_dataset("diagnostics", shape=((config.paths + config.batch_size - 1) // config.batch_size, 20, 4), dtype="f8")
                stream["diagnostics"][batch_index] = diagnostic
                completed += count
                stream.attrs["completed_paths"] = completed
                stream.flush()
                progress.update(stage="batch-committed", completed=completed)
                report = snapshot(diagnostic)
                write_json(directory / f"{split}_progress.json", report)
                print(json.dumps(dict(event="paired_batch", split=split, completed=completed, total=config.paths,
                    elapsed_seconds=time.monotonic() - started)), flush=True)
            report = snapshot(np.asarray(stream["diagnostics"][(config.paths - 1) // config.batch_size]))
            write_json(directory / f"{split}_report.json", report | {"complete": True})
        return target
    finally:
        done.set()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--estimate", type=Path, required=True)
    parser.add_argument("--factor-law", type=Path, required=True)
    parser.add_argument("--paths", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    args = parser.parse_args()
    arrays, metadata = load_legacy_ar1_model(history=args.history, estimate=args.estimate,
        factor_law=args.factor_law, origin="2025-08-29", backend=args.backend)
    config = ForwardScenarioConfig(paths=args.paths, batch_size=args.batch_size, substeps=8,
        seed=2026102801, backend=args.backend, linear_solver="pcr", contiguous_storage=True)
    simulator = PairedSimulator(arrays, config)
    for split in ("train", "validation"):
        generate_paired(arrays=arrays, metadata=metadata, directory=args.run_directory, split=split, config=config,
            simulator=simulator)


if __name__ == "__main__":
    main()
