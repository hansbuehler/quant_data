"""Regenerate the missing first diffusion surface forecast, without fitting."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from quantdata.hedging.model_audit.inference import checkpoint_payload
from quantdata.hedging.model_audit.run import GRID, _record, build_history
from quantdata.sdlv.daily.front_spot.hedging.data.generative_surfaces import SanosFixedGridHistory
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import PublicHedgingPanel
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.config import DiffusionModelConfig
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.data import (
    diffusion_live_context_from_history,
)
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.hedger import (
    sample_diffusion_transitions,
)
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.model import (
    SequentialVolatilityDenoiser,
)
from quantdata.torch import BaseEnvironment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ["QUANTDATA_ROOT"]))
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    day = date(2024, 4, 1)
    destination = arguments.output / "regenerated/AD-Seq-Vol/2024-04-01.npz"
    if destination.exists():
        return
    record = arguments.root / f"experiments/hedging/short-end-regeneration-20261024/diffusion-base-rolling-fixed-v2/{day}/records/{day}.json"
    diagnostics = _record(record)["diagnostics"]
    checkpoint = Path(diagnostics["fit_checkpoint_path"])
    saved = checkpoint_payload(checkpoint, expected_sha256=diagnostics["fit_checkpoint_sha256"], audit_directory=arguments.output)
    config = DiffusionModelConfig(label="diffusion-base-short-end-nms35-2022", grid=GRID)
    if saved["model_code_version"] != config.version.unique_id48 or saved["model_user_version"] != config.user_version or saved["training_end_date"] != str(day):
        raise ValueError("Saved diffusion checkpoint version/date mismatch; do not retrain")
    env = BaseEnvironment(precision=32, device="cpu")
    weights = saved["ema_model"]
    model = SequentialVolatilityDenoiser(env, config=config,
        channel_mean=weights["channel_mean"], channel_scale=weights["channel_scale"])
    model.load_state_dict(weights)
    model.eval()
    panel = PublicHedgingPanel(arguments.root / "datasets/hedging/spx-2022-2025-hedging-public-candidate-v1")
    history = build_history(panel, arguments.output)
    index = history.dates.index(day)
    window = SanosFixedGridHistory(dates=history.dates[:index + 1],
        log_implied_volatilities=history.log_implied_volatilities[:index + 1],
        implied_vol_is_interior=history.implied_vol_is_interior[:index + 1],
        forward_log_returns_into_date=history.forward_log_returns_into_date[:index + 1],
        sessions_from_previous_observation=history.sessions_from_previous_observation[:index + 1])
    context = diffusion_live_context_from_history(window, as_of_date=day, forecast_date=panel.next_one_session_date(day))
    returns, surfaces = [], []
    started = time.monotonic()
    for batch in range(8):
        sample = sample_diffusion_transitions(SimpleNamespace(ema_model=model),
            observed_sequence=context.sequence, path_count=64, seed=20_261_111 + batch)
        returns.append(sample.forward_log_returns)
        surfaces.append(sample.next_log_implied_volatilities)
        elapsed = time.monotonic() - started
        print(f"diffusion recovery {64*(batch+1)}/512 paths elapsed={elapsed:.1f}s remaining={elapsed/(batch+1)*(7-batch):.1f}s", flush=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"origin_date": str(day), "checkpoint": str(checkpoint), "checkpoint_sha256": diagnostics["fit_checkpoint_sha256"],
        "path_count": 512, "batch_size": 64, "batch_seeds": list(range(20_261_111, 20_261_119)),
        "note": "new CPU inference only, source-fixed normalization and saved EMA weights; not the missing original 10000-path forecast"}
    np.savez_compressed(destination, forward_log_returns=np.concatenate(returns),
                        next_log_implied_volatilities=np.concatenate(surfaces), metadata_json=json.dumps(metadata))


if __name__ == "__main__":
    main()
