"""Q scenarios and recurrent hedges for a fixed-book forward-start straddle."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import signal
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from cdxcore.err import verify_inp
from cdxcore.version import version
from filelock import FileLock

from quantdata.gromm2.dlv import SparseTransport, prepare_sparse_transports
from quantdata.gromm2.dynamics import interpolate_loading_expiries
from quantdata.gromm2.rolling_audit import (
    ResourceHeadroomError,
    check_resource_headroom,
    paired_samples,
)
from quantdata.gromm2.rolling_calibration import write_json
from quantdata.gromm2.runtime import configure_backend, gpu_preflight
from quantdata.gromm2.simulation import SimulationState, one_day_risk_neutral_step


@version("0.5")
@dataclass(frozen=True, kw_only=True)
class ForwardScenarioConfig:
    """Sampling plan; times are business days and prices have inception spot one."""

    paths: int = 100_000
    batch_size: int = 256
    substeps: int = 8
    seed: int = 20261027
    fixing_day: int = 10
    terminal_day: int = 20
    density_loading_floor: float = 1.0e-4
    backend: str = "cpu"
    linear_solver: str = "thomas"

    def __post_init__(self) -> None:
        verify_inp(self.paths >= 4 and self.paths % 2 == 0, lambda: "paths must contain complete antithetic pairs")
        verify_inp(self.batch_size >= 2 and self.batch_size % 2 == 0, lambda: "batch_size must be positive and even")
        verify_inp(self.substeps > 0, lambda: "substeps must be positive")
        verify_inp(0 < self.fixing_day < self.terminal_day <= 20, lambda: "invalid forward-start dates")
        verify_inp(0.0 < self.density_loading_floor < 1.0, lambda: "invalid density loading floor")
        verify_inp(self.backend in {"cpu", "gpu"}, lambda: "backend must be cpu or gpu")
        verify_inp(self.linear_solver in {"thomas", "pcr"}, lambda: "linear_solver must be thomas or pcr")


def instrument_book() -> dict[str, np.ndarray]:
    """Freeze one OTM side per inception strike, plus two 20-day wings."""
    contracts = [(expiry, strike) for expiry in (5, 10, 15, 20) for strike in (0.98, 0.99, 1.0, 1.01, 1.02)]
    contracts.extend([(20, 0.95), (20, 1.05)])
    contracts.sort()
    expiries, strikes = map(np.asarray, zip(*contracts, strict=True))
    return {"option_expiry_days": expiries.astype(np.int32), "option_strikes": strikes,
            "option_is_call": strikes >= 1.0}


def forward_payoff(spots: np.ndarray, *, fixing_day: int, terminal_day: int) -> np.ndarray:
    """Return |S_terminal/S_fixing-1|, never an absolute cash-strike payoff."""
    if spots.ndim != 2 or spots.shape[1] <= terminal_day or not np.all(np.isfinite(spots)) or np.any(spots <= 0.0):
        raise ValueError("spot paths must be finite, positive and contain both fixing dates")
    return np.abs(spots[:, terminal_day] / spots[:, fixing_day] - 1.0)


def independent_normal_block(*, config: ForwardScenarioConfig, split: str, completed: int, count: int, factors: int) -> tuple[np.ndarray, np.ndarray]:
    """Use disjoint RNG namespaces for fitting and untouched validation paths."""
    if split not in {"train", "validation"} or count % 2:
        raise ValueError("invalid scenario split or incomplete antithetic pair")
    namespace = 0 if split == "train" else 1
    generator = np.random.default_rng(np.random.SeedSequence([config.seed, namespace, completed]))
    half = generator.standard_normal((config.terminal_day, config.substeps, count // 2, factors))
    factors_normal = np.stack((half, -half), axis=3).reshape(config.terminal_day, config.substeps, count, factors)
    half_background = generator.standard_normal((config.terminal_day, count // 2))
    background_normal = np.stack((half_background, -half_background), axis=2).reshape(config.terminal_day, count)
    return factors_normal, background_normal


def price_fixed_book(*, supports: jax.Array, densities: jax.Array, background: jax.Array,
                     spot: jax.Array, day: jax.Array, book: dict[str, jax.Array], front_day: jax.Array | None = None) -> jax.Array:
    """Price each fixed contract, including its cash settlement, on its own law."""
    remaining = book["option_expiry_days"] - day
    support_origin = day if front_day is None else front_day
    indices = jnp.clip(book["option_expiry_days"] - support_origin - 1, 0, len(supports) - 1)
    support = supports[indices]
    density = densities[:, indices]
    latent = background[:, None, None] * support[None]
    strikes = book["option_strikes"][None, :, None]
    deviation = book["background_volatility"] * jnp.sqrt(jnp.maximum(remaining, 1) / 252.0)
    safe_deviation = jnp.maximum(deviation[None, :, None], jnp.finfo(latent.dtype).eps)
    d1 = jnp.log(latent / strikes) / safe_deviation + 0.5 * safe_deviation
    d2 = d1 - safe_deviation
    calls = latent * jsp.special.ndtr(d1) - strikes * jsp.special.ndtr(d2)
    puts = strikes * jsp.special.ndtr(-d2) - latent * jsp.special.ndtr(-d1)
    kernels = jnp.where(book["option_is_call"][None, :, None], calls, puts)
    options = jnp.sum(jnp.maximum(kernels, 0.0) * density, axis=-1)
    signed_intrinsic = jnp.where(book["option_is_call"][None], spot[:, None] - strikes[..., 0], strikes[..., 0] - spot[:, None])
    options = jnp.where(remaining[None] == 0, jnp.maximum(signed_intrinsic, 0.0), options)
    options = jnp.where(remaining[None] < 0, 0.0, options)
    return jnp.concatenate((spot[:, None], options), axis=-1)


def scenario_function(*, supports: np.ndarray, transports: np.ndarray, expiries_y: np.ndarray,
                      dlvsq: np.ndarray, densities: np.ndarray, loadings: np.ndarray,
                      background_volatility: float, config: ForwardScenarioConfig,
                      ar1_parameters: dict[str, np.ndarray | float] | None = None,
                      loading_reference_days: np.ndarray | None = None):
    """Keep a padded active tower so all twenty daily steps share one JIT graph."""
    dtes = np.rint(expiries_y * 252.0).astype(int)
    if not np.array_equal(dtes[:config.terminal_day], np.arange(1, config.terminal_day + 1)):
        raise ValueError("daily model expiries are required through the payoff maturity")
    if loadings.shape[:2] != dlvsq[1:].shape:
        raise ValueError("Q loadings do not match the initial DLV tower")
    book_numpy = instrument_book()
    book = {name: jnp.asarray(value) for name, value in book_numpy.items()}
    book["background_volatility"] = jnp.asarray(background_volatility)
    basis, _ = np.linalg.qr(loadings.reshape(-1, loadings.shape[-1]))
    basis = np.concatenate((np.zeros((1, dlvsq.shape[1], basis.shape[-1])), basis.reshape(loadings.shape)), axis=0)
    supports_initial = jnp.asarray(supports)
    transports_initial = prepare_sparse_transports(transports)
    identity_transport = prepare_sparse_transports(np.eye(supports.shape[1], dtype=supports.dtype))
    initial_log_dlvsq = jnp.log(jnp.asarray(dlvsq))
    initial_basis = jnp.asarray(basis)
    initial_intervals = jnp.asarray(np.diff(expiries_y))
    loadings_initial = jnp.asarray(loadings)
    if loading_reference_days is not None:
        daily_loadings = []
        for day in range(config.terminal_day):
            active = interpolate_loading_expiries(loadings, loading_reference_days, dtes[day + 1:] - day)
            daily_loadings.append(np.pad(active, ((0, day), (0, 0), (0, 0))))
        floating_loadings = jnp.asarray(np.stack(daily_loadings))
    else:
        floating_loadings = None
    ar1 = None if ar1_parameters is None else {name: jnp.asarray(value) for name, value in ar1_parameters.items()}
    initial_spot = float(densities[0] @ supports[0])
    if abs(initial_spot - 1.0) > 1.0e-8:
        raise ValueError("the normalized initial model forward must equal one")

    def features(state, support, reference_log, projection, cumulative, fixing, day, prices, ar1_state, factor_scales):
        live_projection = projection.at[0].set(0.0)
        live_log = jnp.log(state.dlvsq)
        coordinates = jnp.einsum("bjn,jnf->bf", live_log - reference_log, live_projection)
        maturity_means = jnp.where(jnp.arange(len(support)) < len(support) - day, jnp.mean(live_log, axis=-1), 0.0)
        return jnp.concatenate((
            state.front_density, coordinates, maturity_means, cumulative,
            jnp.log(state.background_level)[:, None], jnp.log(prices[:, 0])[:, None],
            jnp.where(day >= config.fixing_day, jnp.log(fixing), 0.0)[:, None],
            jnp.full((len(prices), 1), day / config.terminal_day),
            jnp.full((len(prices), 1), day >= config.fixing_day), prices[:, 1:],
            jnp.broadcast_to((book["option_expiry_days"] > day)[None], prices[:, 1:].shape),
            ar1_state, factor_scales if ar1 is not None else factor_scales[:, :0],
        ), axis=-1)

    @jax.jit
    def simulate(normals, background_normals):
        paths = normals.shape[2]
        initial_state = SimulationState(jnp.broadcast_to(jnp.asarray(densities[0]), (paths, supports.shape[1])),
                                        jnp.broadcast_to(jnp.asarray(dlvsq), (paths,) + dlvsq.shape), jnp.ones(paths))
        initial_prices = price_fixed_book(supports=supports_initial, densities=jnp.broadcast_to(jnp.asarray(densities), (paths,) + densities.shape),
                                          background=initial_state.background_level, spot=jnp.ones(paths), day=jnp.asarray(0), book=book)
        initial_carry = (initial_state, supports_initial, transports_initial, initial_intervals, loadings_initial,
                         initial_log_dlvsq, initial_basis, jnp.zeros((paths, loadings.shape[-1])), jnp.ones(paths), initial_prices,
                         jnp.broadcast_to(ar1["terminal_log_squared"] if ar1 is not None else jnp.zeros(0),
                                          (paths, loadings.shape[-1] if ar1 is not None else 0)))

        def step(carry, inputs):
            state, support, transport, intervals, active_loadings, reference_log, projection, cumulative, fixing, prices, ar1_state = carry
            day, factor_normal, background_normal = inputs
            if ar1 is None:
                factor_scales = jnp.ones((paths, loadings.shape[-1]))
            else:
                predicted = ar1["variance_intercept"] + ar1["variance_slope"] * ar1_state
                factor_scales = ar1["fixed_factor_scales"] * ar1["scale_calibration"] * jnp.exp(
                    jnp.clip(0.5 * predicted, -ar1["log_scale_bound"], ar1["log_scale_bound"]))
            observed = features(state, support, reference_log, projection, cumulative, fixing, day, prices, ar1_state, factor_scales)
            base_loadings = active_loadings if floating_loadings is None else floating_loadings[day]
            effective_loadings = base_loadings[None] * factor_scales[:, None, None, :]
            result = one_day_risk_neutral_step(support, transport, intervals, state, effective_loadings, factor_normal, background_normal,
                remaining_front_days=jnp.asarray(1), substeps=config.substeps, background_annual_volatility=background_volatility,
                density_loading_floor=config.density_loading_floor, linear_solver=config.linear_solver)
            next_prices = price_fixed_book(supports=support, densities=result.density_tower, background=result.state.background_level,
                                           spot=result.spot, day=day + 1, front_day=day, book=book)
            next_fixing = jnp.where(day + 1 == config.fixing_day, result.spot, fixing)
            next_state = SimulationState(result.state.front_density,
                jnp.concatenate((result.state.dlvsq[:, 1:], result.state.dlvsq[:, -1:]), axis=1), result.state.background_level)
            next_support = jnp.concatenate((support[1:], support[-1:]))
            next_transport = SparseTransport(
                jnp.concatenate((transport.indices[1:], identity_transport.indices[None])),
                jnp.concatenate((transport.weights[1:], identity_transport.weights[None])))
            next_intervals = jnp.concatenate((intervals[1:], intervals[-1:]))
            next_loadings = jnp.concatenate((active_loadings[1:], jnp.zeros_like(active_loadings[-1:])))
            next_reference = jnp.concatenate((reference_log[1:], reference_log[-1:]))
            next_projection = jnp.concatenate((projection[1:], jnp.zeros_like(projection[-1:])))
            mass_error = jnp.max(jnp.abs(jnp.sum(result.density_tower, axis=-1) - 1.0))
            mean_error = jnp.max(jnp.abs(jnp.sum(result.density_tower * support, axis=-1) - result.latent_spot[:, None]))
            minimum_mass = jnp.min(result.density_tower)
            diagnostics = jnp.stack((result.drift_taming_fraction, mass_error, mean_error, minimum_mass))
            next_ar1_state = ar1_state if ar1 is None else jnp.log(
                (factor_scales * result.daily_factor_score / ar1["fixed_factor_scales"]) ** 2 + ar1["squared_score_floor"])
            next_carry = (next_state, next_support, next_transport, next_intervals, next_loadings, next_reference,
                          next_projection, cumulative + factor_normal.sum(axis=0) / jnp.sqrt(252.0 * config.substeps), next_fixing, next_prices,
                          next_ar1_state)
            return next_carry, (observed, next_prices, diagnostics)

        final, (observations, prices, diagnostics) = jax.lax.scan(step, initial_carry, (jnp.arange(config.terminal_day), normals, background_normals))
        payoff = jnp.abs(prices[-1, :, 0] / final[8] - 1.0)
        full_prices = jnp.concatenate((initial_prices[None], prices))
        return jnp.swapaxes(observations, 0, 1), jnp.swapaxes(full_prices, 0, 1), payoff, diagnostics

    return simulate, book_numpy


def sample_summary(values: np.ndarray) -> dict[str, float | int]:
    pairs = paired_samples(np.asarray(values))
    standard_error = float(np.std(pairs, ddof=1) / np.sqrt(len(pairs)))
    mean = float(np.mean(pairs))
    return {"mean": mean, "standard_error": standard_error, "ci95_lower": mean - 1.96 * standard_error,
            "ci95_upper": mean + 1.96 * standard_error, "paths": len(values), "independent_pairs": len(pairs)}


def unconditional_price_diagnostics(prices: np.ndarray, book: dict[str, np.ndarray]) -> dict[str, object]:
    """Test frozen contracts' daily gains and settlement gains using independent pairs."""
    days = prices.shape[1] - 1
    valid = np.column_stack((np.ones(days, dtype=bool), np.arange(1, days + 1)[:, None] <= book["option_expiry_days"][None]))
    increments = np.diff(prices, axis=1)

    def screen(values: np.ndarray, included: np.ndarray) -> dict[str, object]:
        pairs = 0.5 * (values[::2] + values[1::2])
        mean = np.mean(pairs, axis=0)
        error = np.std(pairs, axis=0, ddof=1) / np.sqrt(len(pairs))
        statistic = np.divide(mean, error, out=np.zeros_like(mean), where=error > 0.0)
        absolute = np.abs(statistic[included])
        coordinates = np.argwhere(included)
        worst = np.argsort(absolute)[-10:][::-1]
        return {"independent_pairs": len(pairs), "coordinate_count": len(absolute),
                "maximum_absolute_statistic": float(np.max(absolute)), "percentile_99_absolute_statistic": float(np.quantile(absolute, 0.99)),
                "count_above_1_96": int(np.sum(absolute > 1.96)), "count_above_2_576": int(np.sum(absolute > 2.576)),
                "nonzero_deterministic_coordinates": int(np.sum(included & (error == 0.0) & (mean != 0.0))),
                "worst_coordinates": [{"index": coordinates[index].tolist(), "mean": float(mean[tuple(coordinates[index])]),
                                       "standard_error": float(error[tuple(coordinates[index])]),
                                       "statistic": float(statistic[tuple(coordinates[index])])} for index in worst]}

    settled = np.asarray(book["option_expiry_days"]) <= days
    contracts = np.flatnonzero(settled)
    expiry_gains = prices[:, book["option_expiry_days"][contracts], contracts + 1] - prices[:, 0, contracts + 1]
    return {"interpretation": "Pointwise unconditional Gaussian screens, not a simultaneous test or proof of conditional martingality. Daily indices are zero-based [day, instrument], spot instrument zero. Expiry indices follow the saved option book. Expired positions are excluded; cash settlement is included once.",
            "raw_daily_gains": screen(increments, valid),
            "spot_scaled_daily_gains": screen(increments / prices[:, :-1, :1], valid),
            "option_to_expiry_gains": screen(expiry_gains, np.ones(len(contracts), dtype=bool))}


def load_model(calibration_root: Path, origin: str) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Require a completed, quality-gated Q law; never substitute another model."""
    own = calibration_root / "own_dates" / origin
    rolling = calibration_root / "rolling" / origin
    if not (rolling / "report.json").is_file() or not (rolling / "ar1.npz").is_file():
        raise FileNotFoundError("the requested rolling model has no completed Q diffusion estimate")
    report = json.loads((rolling / "report.json").read_text())
    if not report["historical_calibration"]["ready"]:
        raise ValueError("the rolling model's historical calibration quality gate did not pass")
    own_report = json.loads((own / "report.json").read_text())
    paths = (own / "inputs.npz", own / "fit.npz", rolling / "ar1.npz")
    with np.load(paths[0], allow_pickle=False) as inputs, np.load(paths[1], allow_pickle=False) as state, np.load(paths[2], allow_pickle=False) as law:
        arrays = {"supports": np.asarray(inputs["supports"]), "transports": np.asarray(inputs["transports"]),
                  "expiries_y": np.asarray(inputs["model_expiries_y"]), "dlvsq": np.asarray(state["dlvsq"]),
                  "densities": np.asarray(state["densities"]), "loadings": np.asarray(law["q_log_dlvsq_loadings"])}
    metadata = {"origin": origin, "calibration_root": str(calibration_root.resolve()), "measure": "Q",
                "background_volatility": own_report["config"]["geometry"]["backbone_annual_volatility"],
                "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}
    return arrays, metadata


def generate_scenarios(*, calibration_root: Path | None, origin: str, run_directory: Path, config: ForwardScenarioConfig, split: str,
                       model_inputs: tuple[dict[str, object], dict[str, object]] | None = None) -> Path:
    """Write resumable path-major scenarios outside Git, with pair-level pricing SE."""
    check_resource_headroom()
    configure_backend(config.backend)
    if model_inputs is None:
        if calibration_root is None:
            raise ValueError("calibration_root or explicitly labelled saved model inputs are required")
        arrays, metadata = load_model(calibration_root, origin)
    else:
        arrays, metadata = model_inputs
    software = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in ("forward_hedging.py", "simulation.py", "dlv.py", "bridge.py", "dynamics.py", "runtime.py", "tridiagonal.py")}
    software["quantdata/jax/tridiagonal.py"] = hashlib.sha256((Path(__file__).parents[1] / "jax" / "tridiagonal.py").read_bytes()).hexdigest()
    if model_inputs is not None:
        software["legacy_forward_model.py"] = hashlib.sha256(Path(__file__).with_name("legacy_forward_model.py").read_bytes()).hexdigest()
    identity = json.dumps({"model": metadata, "config": asdict(config), "split": split,
                           "version": config.version.unique_id48, "software_sha256": software}, sort_keys=True)
    run_directory.mkdir(parents=True, exist_ok=True)
    target = run_directory / f"{split}.h5"
    if target.is_file():
        with h5py.File(target, "r") as saved:
            if saved.attrs.get("identity") != identity:
                raise ValueError("scenario file belongs to another model or sampling plan")
            if int(saved.attrs.get("completed_paths", 0)) == config.paths and (run_directory / f"{split}_scenarios.json").is_file():
                return target
    simulate, book = scenario_function(**arrays, background_volatility=metadata["background_volatility"], config=config)
    stopped = threading.Event()
    previous_handlers = {sig: signal.signal(sig, lambda *_: stopped.set()) for sig in (signal.SIGTERM, signal.SIGINT)}
    started, last_progress = time.monotonic(), time.monotonic()
    done = threading.Event()
    progress = {"stage": "initializing", "completed": 0, "total": config.paths}

    def heartbeat() -> None:
        while not done.wait(60):
            print(json.dumps({"event": "forward_heartbeat", "split": split, **progress,
                              "elapsed_seconds": time.monotonic() - started}), flush=True)
            try:
                check_resource_headroom()
            except ResourceHeadroomError:
                stopped.set()
                if not done.wait(60):
                    print(json.dumps({"event": "forced_exit", "message": "Resource stop deadline exceeded; use prior flushed batch."}), flush=True)
                    os.kill(os.getpid(), signal.SIGKILL)
                return

    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        with FileLock(str(target) + ".lock", timeout=0), h5py.File(target, "a") as stream:
            if "identity" in stream.attrs and stream.attrs["identity"] != identity:
                raise ValueError("scenario file belongs to another model or sampling plan")
            stream.attrs["identity"] = identity
            completed = int(stream.attrs.get("completed_paths", 0))
            initial_completed = completed
            progress.update(completed=completed)
            while completed < config.paths:
                count = min(config.batch_size, config.paths - completed)
                factors, background = independent_normal_block(config=config, split=split, completed=completed, count=count, factors=arrays["loadings"].shape[-1])
                progress["stage"] = "compiling_or_simulating"
                observed, prices, payoff, diagnostics = map(np.asarray, simulate(jnp.asarray(factors), jnp.asarray(background)))
                if not all(np.all(np.isfinite(value)) for value in (observed, prices, payoff, diagnostics)):
                    raise FloatingPointError("non-finite Q scenario; no bad paths are silently discarded")
                if np.max(diagnostics[:, 1:3]) > 1.0e-8 or np.min(diagnostics[:, 3]) < -1.0e-12:
                    raise FloatingPointError("Q scenario failed density mass, mean or positivity checks")
                if "features" not in stream:
                    for name, value in (("features", observed), ("prices", prices), ("payoff", payoff)):
                        stream.create_dataset(name, shape=(config.paths,) + value.shape[1:], dtype="f4" if name == "features" else "f8", chunks=(min(config.batch_size, config.paths),) + value.shape[1:])
                    stream.create_dataset("diagnostics", shape=(config.paths // 2, config.terminal_day, 4), dtype="f8")
                    for name, value in book.items():
                        stream.create_dataset(name, data=value)
                    stream.attrs["latent_inputs"] = "Live front density; current log-DLV spatial factor projections and maturity means; cumulative past Brownian factors; Black background; saved AR1 log-squared score state and predictable factor scales when present. No future shocks."
                    stream.attrs["dummy_terminal_features"] = "Appended padding rows have zero projected coordinates and zero maturity means."
                for name, value in (("features", observed), ("prices", prices), ("payoff", payoff)):
                    stream[name][completed:completed + count] = value
                stream["diagnostics"][completed // 2:(completed + count) // 2] = np.broadcast_to(diagnostics, (count // 2,) + diagnostics.shape)
                completed += count
                progress.update(stage="writing_complete_batch", completed=completed)
                stream.attrs["completed_paths"] = completed
                stream.flush()
                if completed >= 4 and (completed == count or completed == config.paths or completed // 4096 > (completed - count) // 4096):
                    write_json(run_directory / f"{split}_progress.json", {
                        "stage": "provisional_path_prefix_not_final_results", "completed_paths": completed, "total_paths": config.paths,
                        "price": sample_summary(stream["payoff"][:completed]),
                        "unconditional_martingale_diagnostics": unconditional_price_diagnostics(stream["prices"][:completed], book),
                        "linear_solver": config.linear_solver, "backend": config.backend})
                elapsed = time.monotonic() - started
                if time.monotonic() - last_progress >= 60 or completed == count or completed == config.paths:
                    check_resource_headroom()
                    rate = (completed - initial_completed) / max(elapsed, 1.0e-9)
                    print(json.dumps({"event": "forward_scenarios", "split": split, "completed": completed, "total": config.paths,
                                      "elapsed_seconds": elapsed, "remaining_seconds": (config.paths - completed) / max(rate, 1.0e-9),
                                      "maximum_drift_taming_fraction": float(np.max(diagnostics[:, 0]))}), flush=True)
                    last_progress = time.monotonic()
                if stopped.is_set():
                    raise InterruptedError("forward scenarios checkpointed at a complete path batch")
            spot_paths = stream["prices"][:, :, 0]
            report = {"model": metadata, "config": asdict(config), "split": split, "price": sample_summary(stream["payoff"][:]),
                      "payoff": f"abs(S_{config.terminal_day}/S_{config.fixing_day}-1)", "instrument_count_including_spot": 1 + len(book["option_strikes"]),
                      "terminal_spot_range": [float(np.min(spot_paths[:, -1])), float(np.max(spot_paths[:, -1]))],
                      "spot_exploration": {"fraction_ever_below_0_95": float(np.mean(np.any(spot_paths < 0.95, axis=1))),
                                           "fraction_ever_above_1_05": float(np.mean(np.any(spot_paths > 1.05, axis=1))),
                                           "minimum": float(np.min(spot_paths)), "maximum": float(np.max(spot_paths))}}
            report["unconditional_martingale_diagnostics"] = unconditional_price_diagnostics(stream["prices"][:], book)
            write_json(run_directory / f"{split}_scenarios.json", report)
    finally:
        done.set()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--calibration-root", type=Path)
    source.add_argument("--legacy-history", type=Path)
    parser.add_argument("--legacy-estimate", type=Path)
    parser.add_argument("--legacy-factor-law", type=Path)
    parser.add_argument("--origin", default="2025-08-29")
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--paths", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--linear-solver", choices=("thomas", "pcr"), default="thomas")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--scenarios-only", action="store_true")
    arguments = parser.parse_args()
    configure_backend(arguments.backend)
    if arguments.backend == "gpu":
        preflight = gpu_preflight()
        write_json(arguments.run_directory / "gpu_preflight.json", preflight)
        print(json.dumps({"event": "allocated_gpu_preflight", **preflight}), flush=True)
    model_inputs = None
    if arguments.legacy_history is not None:
        if arguments.wait or arguments.legacy_estimate is None or arguments.legacy_factor_law is None:
            parser.error("legacy AR1 requires both estimate and factor-law files, without --wait")
        from quantdata.gromm2.legacy_forward_model import load_legacy_ar1_model

        jax.config.update("jax_enable_x64", True)
        model_inputs = load_legacy_ar1_model(history=arguments.legacy_history, estimate=arguments.legacy_estimate,
                                             factor_law=arguments.legacy_factor_law, origin=arguments.origin, backend=arguments.backend)
    elif arguments.legacy_estimate is not None or arguments.legacy_factor_law is not None:
        parser.error("legacy estimate and factor law require --legacy-history")
    if arguments.wait:
        model_directory = arguments.calibration_root / "rolling" / arguments.origin
        while not all((model_directory / name).is_file() for name in ("ar1.npz", "report.json")):
            check_resource_headroom()
            print(json.dumps({"event": "waiting_for_quality_gated_q_model", "origin": arguments.origin}), flush=True)
            time.sleep(60)
    config = ForwardScenarioConfig(paths=arguments.paths, batch_size=arguments.batch_size, substeps=arguments.substeps,
                                   backend=arguments.backend, linear_solver=arguments.linear_solver)
    for split in ("train", "validation"):
        generate_scenarios(calibration_root=arguments.calibration_root, origin=arguments.origin,
                           run_directory=arguments.run_directory, config=config, split=split, model_inputs=model_inputs)
    if not arguments.scenarios_only:
        from quantdata.gromm2.hedging_policy import TrainConfig, run_hedging_experiment

        jax.clear_caches()
        gc.collect()
        run_hedging_experiment(arguments.run_directory, train_config=TrainConfig(device="cuda" if arguments.backend == "gpu" else "cpu"))


if __name__ == "__main__":
    main()
