from __future__ import annotations

from dataclasses import asdict

import pytest

from quantdata.gromm2.benchmark_combined_pipeline import (
    benchmark,
    checkpoint_configuration,
    pipeline_recommendation,
    summarize_timings,
)
from quantdata.gromm2.combined_hedging import CombinedBookConfig
from quantdata.gromm2.market_books import primary_contracts
from quantdata.torch.ffw import FeedForwardConfig


def test_checkpoint_configuration_preserves_combined_geometry():
    book = primary_contracts()
    config = CombinedBookConfig(label="benchmark-smoke", feature_count=3, hidden_size=4,
        option_expiry_days=tuple(map(int, book["expiries"])) + (20,) * 3,
        option_start_days=tuple(map(int, book["starts"])) + (0,) * 3,
        locked=tuple(map(bool, book["locked"])), call_sides=tuple(map(bool, book["calls"])),
        basket_weights=((0.5,), (-0.5,)), deep=FeedForwardConfig(num_depth=1, num_width=4))
    saved = dict(identity=dict(config=asdict(config)))
    restored = checkpoint_configuration(saved)
    assert asdict(restored) == asdict(config)
    assert isinstance(saved["identity"]["config"]["deep"], dict)


@pytest.mark.parametrize("values", [[], [float("nan")], [float("inf")], [-1.0]])
def test_timing_summary_rejects_invalid_measurements(values):
    with pytest.raises(ValueError):
        summarize_timings(values)


def test_timing_summary_retains_slow_observations():
    result = summarize_timings([0.001, 0.002, 0.003, 1.0])
    assert result["mean_ms"] == pytest.approx(251.5)
    assert result["median_ms"] == pytest.approx(2.5)
    assert result["maximum_ms"] == 1000.0


def trials(*, resident_seconds=8.0, active=True):
    return [dict(phase=phase, paths=1000, elapsed_seconds=resident_seconds if phase == "resident" else 10.0,
        cache_active_throughout=active) for phase in ("mapped-before-cache", "mapped-with-cache", "resident",
            "resident", "mapped-with-cache", "mapped-after-release")]


@pytest.mark.parametrize("resident_seconds,recommended", [(8.0, True), (9.9, False), (12.0, False)])
def test_recommendation_requires_controlled_material_throughput_improvement(resident_seconds, recommended):
    result = pipeline_recommendation(trials(resident_seconds=resident_seconds))
    assert result["mapped_paths_per_second"] == pytest.approx(100.0)
    assert result["resident_paths_per_second"] == pytest.approx(1000 / resident_seconds)
    assert result["recommendations"]["host_cache_training"] is recommended


@pytest.mark.parametrize("records", [[], trials()[:3], trials()[1:-1], trials(active=False)])
def test_incomplete_or_evicted_cache_trials_are_never_recommended(records):
    assert not pipeline_recommendation(records)["recommendations"]["host_cache_training"]


def test_memory_pressured_mapped_trials_cannot_create_a_false_cache_recommendation():
    records = trials(resident_seconds=8)
    for trial in records:
        if trial["phase"] in {"mapped-before-cache", "mapped-after-release"}:
            trial["elapsed_seconds"] = 3
    result = pipeline_recommendation(records)
    assert result["mapped_with_cache_paths_per_second"] == pytest.approx(100)
    assert result["mapped_paths_per_second"] == pytest.approx(1000 / 3)
    assert result["resident_paths_per_second"] == pytest.approx(125)
    assert not result["recommendations"]["host_cache_training"]


def test_benchmark_rejects_unbounded_work_before_loading_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="batches"):
        benchmark(checkpoint=tmp_path / "absent.pt", run_directory=tmp_path,
            output_directory=tmp_path / "output", batches=1000000)


def test_benchmark_requires_external_output_root_before_loading_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("QUANTDATA_ROOT", str(tmp_path / "data"))
    with pytest.raises(ValueError, match="QUANTDATA_ROOT"):
        benchmark(checkpoint=tmp_path / "absent.pt", run_directory=tmp_path,
            output_directory=tmp_path / "not-data" / "output")
