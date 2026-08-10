"""Replay PCA-IV static checks on the actual 7x20 traded strike grids."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from cdxcore.err import verify

from quantdata.dysanos.implied.static_arbitrage import (
    TradedStrikeStaticArbitrageAccumulator,
)

from .matched_cache import MatchedUniverseBlockCache

_DEFAULT_CACHE_ROOT = Path("/media/storage/tmp/quantdata/dysanos/required-runs")
_DEFAULT_REPORT_ROOT = Path(
    "quantdata/dysanos/grid/gendysanos/reports/required/traded-strike-static"
)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _completed_cache(
    cache_root: Path,
    *,
    frequency: str,
    paths: int,
) -> MatchedUniverseBlockCache:
    universe_root = cache_root / "universes"
    matches: list[tuple[Path, dict[str, Any]]] = []
    for manifest_path in sorted(
        universe_root.glob(f"pca-iv-{frequency}-{paths}-*/manifest.json")
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        configuration = manifest.get("configuration", {})
        if (
            manifest.get("status") == "complete"
            and configuration.get("generator") == "pca-iv"
            and configuration.get("trading_frequency") == frequency
            and configuration.get("num_samples") == paths
        ):
            matches.append((manifest_path, manifest))
    verify(
        len(matches) == 1,
        lambda: (
            f"expected one complete PCA-IV {frequency}/{paths} cache under "
            f"{universe_root}, found {[str(path) for path, _ in matches]}"
        ),
    )
    manifest_path, manifest = matches[0]
    return MatchedUniverseBlockCache(
        universe_root,
        cache_uid=manifest["cache_uid"],
        configuration=manifest["configuration"],
        num_samples=manifest["num_samples"],
    )


def run_traded_strike_static(
    *,
    cache_root: Path,
    report_root: Path,
    frequency: str,
    paths: int,
) -> Path:
    """Replay one completed market and write its weighted static report."""

    cache = _completed_cache(cache_root, frequency=frequency, paths=paths)
    accumulator = TradedStrikeStaticArbitrageAccumulator(seed=700_977)
    started = time.perf_counter()
    blocks = 0
    for block in cache.iter_blocks():
        universe = block.expiry
        spot = np.asarray(universe["spot"], dtype=np.float64)
        normalized_calls = (
            np.asarray(universe["C"], dtype=np.float64)
            / spot[:, :, None, None]
        )
        relative_strikes = np.exp(
            np.asarray(universe["log_strikes"], dtype=np.float64)
        )
        accumulator.update(
            normalized_calls,
            strikes=relative_strikes,
            log_path_weights=universe.get("log_path_weight"),
            path_offset=block.start,
        )
        blocks += 1
        if blocks % 10 == 0 or block.stop == paths:
            print(
                f"PCA-IV {frequency} {paths:,}: checked {block.stop:,} paths",
                flush=True,
            )
    report = {
        "generator": "pca-iv",
        "trading_frequency": frequency,
        "num_samples": paths,
        "source_cache": cache.cache_uid,
        "price_normalization": (
            "cached cash call divided by entry spot; strike is cached relative "
            "strike exp(log_strikes)"
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "static_arbitrage": accumulator.report(),
    }
    destination = report_root / f"pca-iv-{frequency}-{paths}.json"
    _atomic_json(destination, report)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Check monotonicity, vertical slope, and convexity on actual "
            "PCA-IV traded strikes"
        )
    )
    parser.add_argument(
        "--frequency", choices=("daily", "weekly", "both"), default="both"
    )
    parser.add_argument(
        "--paths", type=int, nargs="+", default=[100_000, 1_000_000]
    )
    parser.add_argument("--cache-root", type=Path, default=_DEFAULT_CACHE_ROOT)
    parser.add_argument("--report-root", type=Path, default=_DEFAULT_REPORT_ROOT)
    args = parser.parse_args()
    frequencies = (
        ("daily", "weekly") if args.frequency == "both" else (args.frequency,)
    )
    for paths in args.paths:
        for frequency in frequencies:
            destination = run_traded_strike_static(
                cache_root=args.cache_root,
                report_root=args.report_root,
                frequency=frequency,
                paths=paths,
            )
            print(f"wrote {destination}", flush=True)


if __name__ == "__main__":
    main()

