"""Run a saved-market terminal hedge episode, with daily causal re-estimation."""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import threading
from datetime import date
from pathlib import Path

from filelock import FileLock

from quantdata.hedge2.cash import MarketStore
from quantdata.hedge2.source import SourceArchive
from quantdata.hedge2.terminal_market import TerminalSimulationConfig
from quantdata.hedge2.terminal_replay import load_origin_decision, replay_terminal_episode
from quantdata.hedge2.terminal_train import TerminalTrainConfig
from quantdata.torch import BaseEnvironment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-configuration", type=Path, required=True)
    parser.add_argument("--origin", type=date.fromisoformat, required=True)
    parser.add_argument("--claim", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--previous-origin-directory", type=Path)
    parser.add_argument("--objective", choices=("mean_variance", "power_exponential"), default="mean_variance")
    parser.add_argument("--risk-aversion", type=float, default=10.)
    parser.add_argument("--side", type=int, choices=(-1, 1), default=-1)
    parser.add_argument("--paths", type=int)
    parser.add_argument("--maximum-updates", type=int, default=2000)
    parser.add_argument("--minimum-updates", type=int, default=100)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--precision", type=int, choices=(32, 64), default=64)
    parser.add_argument("--diagnostic-only", action="store_true")
    arguments = parser.parse_args()
    configuration_bytes = arguments.source_configuration.read_bytes()
    sources = json.loads(configuration_bytes)
    store = MarketStore(market_directory=Path(sources["market_directory"]),
        licensed_fit_directory=Path(sources["licensed_fit_directory"]),
        raw_quote_directory=Path(sources["raw_quote_directory"]))
    daily_directory = Path(sources["daily_directory"])
    env = BaseEnvironment(device=arguments.device, precision=arguments.precision, debug=True)
    previous = (None if arguments.previous_origin_directory is None else load_origin_decision(
        arguments.previous_origin_directory, env=env, daily_directory=daily_directory))
    stopping = threading.Event()
    handlers = {number: signal.signal(number, lambda signum, frame: stopping.set())
                for number in (signal.SIGINT, signal.SIGTERM)}
    try:
        arguments.output_directory.mkdir(parents=True, exist_ok=True)
        with FileLock(str(arguments.output_directory / "launcher.lock"), timeout=0):
            try:
                result = replay_terminal_episode(origin=arguments.origin, claim_label=arguments.claim, store=store,
                    archive=SourceArchive(root=Path(sources["source_root"])), daily_directory=daily_directory,
                    dynamics_directory=Path(sources["dynamics_directory"]), output_directory=arguments.output_directory,
                    objective=arguments.objective, risk_aversion=arguments.risk_aversion, target_side=arguments.side,
                    env=env, train_config=TerminalTrainConfig(maximum_updates=arguments.maximum_updates,
                        minimum_updates=arguments.minimum_updates), simulation_config=TerminalSimulationConfig(paths=arguments.paths),
                    previous_origin_decision=previous, diagnostic_only=arguments.diagnostic_only, stop_requested=stopping.is_set)
            except Exception as error:
                unavailable = isinstance(error, FileNotFoundError) or str(error).startswith(("Missing observed mids", "No PM"))
                (arguments.output_directory / "status.json").write_text(json.dumps({
                    "status": "market_data_unavailable" if unavailable else "review_required",
                    "error_type": type(error).__name__, "exact_error": str(error),
                    "accepted_records_preserved": True}, indent=2) + "\n")
                raise
            provenance = {"source_configuration": str(arguments.source_configuration.resolve()),
                "source_configuration_sha256": hashlib.sha256(configuration_bytes).hexdigest(),
                "sources": sources, "requested_maximum_updates": arguments.maximum_updates,
                "previous_origin_directory": None if arguments.previous_origin_directory is None else str(arguments.previous_origin_directory.resolve())}
            (arguments.output_directory / "launcher.json").write_text(json.dumps(provenance, indent=2) + "\n")
            print(json.dumps(result.summary, indent=2), flush=True)
    finally:
        for number, handler in handlers.items():
            signal.signal(number, handler)


if __name__ == "__main__":
    main()
