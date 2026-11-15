"""Independent rolling execution using the original quant_data model code."""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock

from quantdata.hedging.model_audit.rolling_dysanos import window_dates
from quantdata.hedging.model_audit.rolling_versions import INITIALIZATION_POLICY, origins
from quantdata.hedging.model_audit.run import _hash, _json


def paper_geometry():
    from quantdata.dysanos.flex.config import (
        FrontTermEtaSharedStrikeCappedUnboundedDlvFlexGeometryConfig,
    )

    return FrontTermEtaSharedStrikeCappedUnboundedDlvFlexGeometryConfig(num_model_expiries=18,
        explicit_model_business_dtes=tuple(range(1, 6)) + tuple(range(10, 71, 5)))


def market_from_cache(day: date, geometry):
    """Core supplies data only; every model and optimizer is the original code."""
    from openqdata.arbfree.arbfree import MkArbFree
    from openqdata.widb.arbfree import mk_arb_free_day

    from quantdata.biz_calendar import NMS
    from quantdata.dysanos.flex.market import FlexMarketDay, MarketEligibilityAudit
    from quantdata.hedge2.source import SourceArchive

    rows = mk_arb_free_day("SPX", day, mk_arb_free=MkArbFree(smoothness=.1), override_cache_mode="cacheonly")
    if rows is None or rows.empty or set(pd.to_datetime(rows.date).dt.date) != {day}:
        raise ValueError(f"Missing or incorrectly dated licensed cache: {day}")
    source_root = Path(os.environ["OQ_DATA_PROCESSED"]).parents[1]
    pricer, provenance = SourceArchive(root=source_root, smoothness=.1).load(day)
    model_atm = np.asarray(pricer.interpolate_atm_vols(geometry.model_expiries_y))
    atm = np.asarray(pricer.interpolate_atm_vols(rows.tte_y.to_numpy()))
    nms = NMS.to_nms(rows.pure_strike.to_numpy(), rows.tte_y.to_numpy(), vol=atm,
        eps_vol=float(model_atm[0]))
    selected = rows[(rows.tte_y > 0.) & (rows.tte_y <= geometry.model_expiries_y[-1] + 1e-8)
        & ~np.isclose(rows.tte_y * 252., .2, atol=1e-4) & ~rows.is_american
        & (nms >= geometry.nms_range[0]) & (nms <= geometry.nms_range[1])].copy()
    books = []
    for side, is_call in (("call", True), ("put", False)):
        book = selected.copy()
        normalized_forward = book.df.to_numpy() * book.fwd.to_numpy()
        book["book_bid"] = book[f"mkt_{side}_bid"].to_numpy() / normalized_forward
        book["book_ask"] = book[f"mkt_{side}_ask"].to_numpy() / normalized_forward
        book["book_is_call"] = is_call
        book["book_fit"] = book.pure_fitX_price + (0. if is_call else book.pure_strike - 1.)
        book["book_volume"] = book[f"mkt_{side}_log_volume"]
        book["book_interest"] = book[f"mkt_{side}_log_open_interest"]
        book["book_id"] = [f"{side}-{day}-{index}" for index in book.index]
        active = np.isfinite(book.book_bid) & np.isfinite(book.book_ask) & (book.book_bid >= 0.) & (book.book_ask > book.book_bid)
        books.append(book[active])
    selected = pd.concat(books, ignore_index=True).sort_values("tte_y", kind="stable")
    unique_times = np.sort(selected.tte_y.unique())
    lookup = {value: index for index, value in enumerate(unique_times)}
    expiry_index = np.asarray([lookup[value] for value in selected.tte_y], dtype=int)
    groups = [selected[selected.tte_y == value] for value in unique_times]
    annual = np.asarray(pricer.interpolate_atm_vols(unique_times))
    bounds = NMS.from_nms(np.asarray(geometry.nms_range)[None], unique_times[:, None], vol=annual[:, None],
        eps_vol=float(model_atm[0]))
    totals = model_atm * np.sqrt(geometry.model_expiries_y)
    count = len(selected)
    return FlexMarketDay(ticker="SPX", trading_date=day, expiries_y=unique_times,
        expiry_dates=tuple(str(pd.Timestamp(group.expiry.iloc[0]).date()) for group in groups),
        expiry_ids=np.asarray([int(group.expiry_id.iloc[0]) for group in groups]),
        expiry_is_am=np.asarray([bool(group.is_am.iloc[0]) for group in groups]), expiry_index=expiry_index,
        strikes=selected.pure_strike.to_numpy(float), is_call=selected.book_is_call.to_numpy(bool),
        bids=selected.book_bid.to_numpy(float), asks=selected.book_ask.to_numpy(float),
        fit_prices=selected.book_fit.to_numpy(float), log_volumes=selected.book_volume.to_numpy(float),
        log_open_interests=selected.book_interest.to_numpy(float), quote_ids=tuple(selected.book_id),
        atm_annual_vols=annual, model_atm_total_vols=totals, strike_lower_bounds=bounds[:, 0], strike_upper_bounds=bounds[:, 1],
        market_uid=f"licensed-smoothness-0.1-{day}-{provenance['source_sha256']}", spot_ref=float(selected.spot.iloc[0]),
        audit=MarketEligibilityAudit(2 * len(rows), 2 * len(rows[rows.tte_y <= geometry.model_expiries_y[-1]]),
            count, rows.tte_y.nunique(), len(unique_times)))


def same_origin_load_config(training, path: Path):
    """Completed same-origin stages are loaded, never initialized from another origin."""
    if not path.exists():
        return training
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    return replace(training, just_load=payload.get("phase") == "complete")


def fit_joint_to_convergence(model, market, *, dates, objective, training, checkpoint: Path,
                             decoder_sha256: str, environment):
    """Continue an exhausted same-origin schedule without weakening its original audit."""
    import torch

    from quantdata.dysanos.flex.joint_direct import _convergence_audit, fit_joint_direct_model
    from quantdata.dysanos.flex.training import TrainingConvergenceError

    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        training = replace(training, num_epochs=max(training.num_epochs, int(payload["train_config"]["num_epochs"])))
    while True:
        try:
            return fit_joint_direct_model(model, market, dates=dates, objective_config=objective,
                train_config=training, checkpoint_path=checkpoint, source_decoder_sha256=decoder_sha256, env=environment)
        except TrainingConvergenceError as error:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            audit = _convergence_audit(payload["history"], config=training)
            completed = int(payload["completed_epochs"])
            if audit.get("reason") != "trailing_window_not_converged" or completed != training.num_epochs:
                raise
            next_epochs = completed + 100
            record = {"exact_warning": str(error), "audit": audit, "completed_epochs": completed,
                "next_audit_epoch": next_epochs, "original_schedule_horizon": payload["schedule_horizon"],
                "convergence_tolerance": training.convergence_relative_tolerance,
                "last_learning_rates": payload["scheduler_state_dict"].get("_last_lr"),
                "policy": "restore same-origin model, optimizer, scheduler and RNG; retain original schedule and acceptance threshold"}
            _json(checkpoint.parent / f"joint-continuations/epoch-{completed}.json", record)
            print(json.dumps(record), flush=True)
            training = replace(training, num_epochs=next_epochs)


class Progress:
    """Lightweight persistent status without synchronizing numerical tensors."""

    def __init__(self, directory: Path, *, family: str, origin: str):
        self.directory, self.family, self.origin = directory, family, origin
        self.started, self.finished = time.monotonic(), threading.Event()
        self.phase = "initializing"
        self.completed, self.total = 0, 4

    def emit(self) -> None:
        event = {"family": self.family, "origin": self.origin, "phase": self.phase,
            "elapsed_seconds": time.monotonic() - self.started,
            "completed_stages": self.completed, "total_stages": self.total, "eta_seconds": None}
        print(json.dumps(event), flush=True)
        _json(self.directory / "progress.json", event)

    def __enter__(self):
        def heartbeat():
            while not self.finished.wait(60.):
                self.emit()
        self.thread = threading.Thread(target=heartbeat, daemon=True)
        self.thread.start()
        self.emit()
        return self

    def __exit__(self, *args):
        self.finished.set()
        self.thread.join(timeout=2.)


def fit_paper_origin(root: Path, reference: Path, *, origin: str, prepare_only: bool = False,
                     device: str = "cpu", fit_only: bool = False, paper_midpoint: bool = False,
                     market_inputs_from: Path | None = None) -> None:
    from quantdata.biz_calendar import get_business_days
    from quantdata.dysanos.flex.config import (
        FlexObjectiveConfig,
        FlexTrainConfig,
        JointDirectDynamicsConfig,
        NativeStateConfig,
        PcaDecoderConfig,
    )
    from quantdata.dysanos.flex.joint_direct import (
        JointDirectTrainConfig,
        initialize_joint_model,
        standardize_affine_direct_model,
    )
    from quantdata.dysanos.flex.model import JointDirectDynamicsModel
    from quantdata.dysanos.flex.training import fit_latent_states, fit_native_states
    from quantdata.dysanos.flex.weekly_validation import exact_weekly_pairs
    from quantdata.torch import BaseEnvironment

    origins(reference).index(origin)
    cutoff = date.fromisoformat(origin)
    dates = window_dates(get_business_days(date(2020, 1, 1), cutoff), cutoff)
    directory = root / f"fits/dysanos1/{origin}"
    directory.mkdir(parents=True, exist_ok=True)
    geometry = paper_geometry()
    with FileLock(directory / ".training.lock"), Progress(directory, family="DYSANOS v1", origin=origin) as progress:
        progress.phase = "read unchanged licensed cache"
        import torch

        market = []
        for index, day in enumerate(dates):
            cache = root / f"market-days/{day}.pt"
            if cache.exists():
                market.append(torch.load(cache, map_location="cpu", weights_only=False))
            else:
                previous = None if market_inputs_from is None else market_inputs_from / f"market-days/{day}.pt"
                raw = torch.load(previous, map_location="cpu", weights_only=False) if previous is not None and previous.exists() else market_from_cache(day, geometry)
                if paper_midpoint:
                    raw = replace(raw, fit_prices=.5 * (raw.bids + raw.asks),
                        market_uid=raw.market_uid + "-observed-midpoint-v1")
                market.append(raw)
                cache.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache.with_suffix(".partial.pt")
                torch.save(market[-1], temporary)
                temporary.replace(cache)
            if paper_midpoint and not np.array_equal(market[-1].fit_prices, .5 * (market[-1].bids + market[-1].asks)):
                raise ValueError("Paper-objective training cannot reuse SANOS-centered market targets")
            if (index + 1) % 25 == 0:
                print(f"market dates completed={index+1}/{len(dates)}", flush=True)
        market = tuple(market)
        protocol = {"origin": origin, "dates": [str(day) for day in dates], "geometry": asdict(geometry),
            "initialization": INITIALIZATION_POLICY, "implementation": "original quantdata.dysanos.flex Torch paper trainer",
            "state_dimension": 200, "transition_sessions": 5, "source_smoothness": .1,
            "targets": "cached separate market call/put bid/ask; source SANOS supplies ATM interpolation only", "training_precision": 32,
            "centering_target": "observed bid/ask midpoint" if paper_midpoint else "cached SANOS fitted price (not paper midpoint)",
            "market_data_reused_from": None if market_inputs_from is None else str(market_inputs_from),
            "training_device": device,
            "market_fingerprints": [day.training_fingerprint for day in market]}
        _json(directory / "PROTOCOL.json", protocol)
        if prepare_only:
            progress.phase = "market input preparation complete"
            progress.emit()
            return
        objective = FlexObjectiveConfig(fit_weight=.05)
        training = FlexTrainConfig(num_epochs=1000, day_batch_size=16, device=device, precision=32,
            checkpoint_every=5, report_every=5, compile_sanos_solver=device != "cpu")
        progress.phase = "native market fit"
        progress.completed = 1
        native = fit_native_states(market, config=NativeStateConfig(label="rolling-paper-native", geometry=geometry),
            objective_config=objective, train_config=same_origin_load_config(training, directory / "native.pt"),
            checkpoint_path=directory / "native.pt")
        progress.phase = "PCA initialization and market-refined affine-200"
        progress.completed = 2
        affine = fit_latent_states(market, native,
            config=PcaDecoderConfig(label="rolling-paper-affine200", state_size=200, geometry=geometry),
            objective_config=objective,
            supervised_train_config=FlexTrainConfig(num_epochs=300, day_batch_size=128, learning_rate=.002,
                device=device, precision=32, checkpoint_every=5, report_every=5),
            market_train_config=same_origin_load_config(FlexTrainConfig(num_epochs=1000, day_batch_size=16,
                learning_rate=.0001, device=device, precision=32, checkpoint_every=5, report_every=5,
                compile_sanos_solver=device != "cpu"), directory / "affine.pt"),
            checkpoint_path=directory / "affine.pt")
        progress.phase = "joint five-session state/decoder/VAR refinement"
        progress.completed = 3
        environment = BaseEnvironment(precision=32, device=device, debug=False)
        model = JointDirectDynamicsModel(environment, num_days=len(dates),
            config=JointDirectDynamicsConfig(label="rolling-paper-joint200", state_size=200, geometry=geometry))
        before, after = exact_weekly_pairs(dates, weeks=1)
        initialize_joint_model(model, standardize_affine_direct_model(affine.model), origin_indices=before, endpoint_indices=after)
        history, diagnostics = fit_joint_to_convergence(model, market, dates=dates, objective=objective,
            training=JointDirectTrainConfig(day_batch_size=16, checkpoint_every=1, report_every=1, seed=9281,
                compile_sanos_solver=device != "cpu"),
            checkpoint=directory / "joint.pt", decoder_sha256=_hash(directory / "affine.pt"), environment=environment)
        _json(directory / "joint-diagnostics.json", diagnostics)
        _json(directory / "joint-history.json", history)
        _json(directory / "FIT_COMPLETE.json", {"family": "DYSANOS v1", "origin": origin,
            "model_sha256": _hash(directory / "joint.pt"), "measure": "P",
            "status": "training complete; one-day covariance embedding and CPU statistics not yet evaluated",
            "centering_target": protocol["centering_target"]})
        if fit_only:
            progress.phase = "fit complete; statistics deferred until all requested training finishes"
            progress.completed = 4
            progress.emit()
            return
        progress.phase = "matched full-state one-day forecast evaluation"
        score_paper_origin(root, reference, origin=origin)
        _json(directory / "COMPLETE.json", {"family": "DYSANOS v1", "origin": origin,
            "model": str(directory / "joint.pt"), "model_sha256": _hash(directory / "joint.pt"),
            "measure": "P", "initialization": INITIALIZATION_POLICY,
            "forecast_statistics": "one-day observable evaluation saved; other diagnostics pending"})
        progress.phase = "complete"
        progress.completed = 4
        progress.emit()


def score_paper_origin(root: Path, reference: Path, *, origin: str, paths: int = 1000) -> None:
    """Use the entire fitted state law, never replace it with diagnostic IV PCs."""
    import torch

    from quantdata.dysanos.flex.presentation_hedging import load_presentation_model
    from quantdata.hedging.model_audit.metrics import covariance
    from quantdata.hedging.model_audit.paper_v1 import daily_law, observable

    _, model = load_presentation_model(root / f"fits/dysanos1/{origin}/joint.pt", device="cpu", precision=64)
    with torch.no_grad():
        transition, factor, embedding = daily_law(model.mean_reversion_generator().numpy(), model.innovation_cholesky().numpy())
        current = model.daily_states[-1].detach().numpy()
        mean = model.equilibrium_mean.detach().numpy()
        forecast = mean + (current - mean) @ transition
        generator = np.random.default_rng(20_261_112)
        shocks = generator.standard_normal((paths, model.config.state_size))
        following = forecast + shocks @ factor.T
        initial_surface = observable(model, torch.as_tensor(current[None], dtype=torch.float64))[0]
        surfaces = np.concatenate([observable(model, torch.as_tensor(block, dtype=torch.float64))
            for block in np.array_split(following, int(np.ceil(paths / 32)))])
    shared = json.loads((reference / f"daily/FuNVol/{origin}.json").read_text())
    changes = surfaces - initial_surface
    record = {name: shared[name] for name in ("origin_date", "endpoint", "actual_change", "actual_spot", "actual_targets")}
    record.update(model="DYSANOS v1-P", status="scored", scenario_count=paths, valid_scenario_count=paths,
        surface_mean=changes.mean(axis=0).tolist(), surface_covariance=covariance(changes, ddof=0).tolist(),
        scenario_covariance_ddof=0, invalid_surface_rate=0.,
        extreme_surface_rate=float(np.any(surfaces > np.log(3.), axis=1).mean()),
        forecast_rmse=float(np.sqrt(np.mean((changes.mean(axis=0) - np.asarray(shared["actual_change"]))**2))),
        diagnostics={"embedding": embedding, "sampling": "full market-refined200-state Gaussian law; no PCA truncation",
            "source_checkpoint": str(root / f"fits/dysanos1/{origin}/joint.pt"),
            "spot_leverage": "not asserted: original paper VAR has no separately fitted joint spot coordinate"},
        bidask={"status": "not_scored", "reason": "Matched quote audit pending"})
    _json(root / f"daily/DYSANOS_v1-P/{origin}.json", record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--origin", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--fit-only", action="store_true")
    parser.add_argument("--paper-midpoint", action="store_true")
    parser.add_argument("--market-inputs-from", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    args = parser.parse_args()
    if args.score_only:
        score_paper_origin(args.root, args.reference, origin=args.origin)
    else:
        fit_paper_origin(args.root, args.reference, origin=args.origin, prepare_only=args.prepare_only,
            device=args.device, fit_only=args.fit_only, paper_midpoint=args.paper_midpoint,
            market_inputs_from=args.market_inputs_from)


if __name__ == "__main__":
    main()
