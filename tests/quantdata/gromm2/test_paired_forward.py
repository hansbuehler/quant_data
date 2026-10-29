from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from quantdata.gromm2 import paired_forward as paired
from quantdata.gromm2.forward_hedging import ForwardScenarioConfig
from quantdata.gromm2.market_books import primary_contracts
from quantdata.gromm2.market_hedging import load_market_scenarios
from quantdata.gromm2.paired_experiment import stages


class TinySimulator:
    calls = 0

    def __init__(self, arrays, config):
        self.primary = primary_contracts()
        self.candidates = {"strikes": np.asarray([1.0]), "expiries": np.asarray([25]), "calls": np.asarray([True])}
        self.weights = [np.ones((1, 1)), np.ones((1, 1))]
        self.basket_reports = [{}, {}]

    def simulate(self, factors, background, progress):
        type(self).calls += 1
        count = factors.shape[2]
        return dict(core_features=np.zeros((count, 20, 4), dtype="f4"), spots=np.ones((2, count, 21)),
            backgrounds=np.ones((2, count, 21)), option_prices=np.zeros((2, count, 21, 84)),
            factor_prices=np.zeros((2, count, 21, 1)), daily_calls=np.ones((2, count, 20)) * 0.001,
            payoff=np.ones((2, count)) * 0.02, diagnostics=np.zeros((20, 4)))


def test_both_backgrounds_share_simulation_and_complete_files_map_without_copies(tmp_path, monkeypatch):
    monkeypatch.setattr(paired, "PairedSimulator", TinySimulator)
    monkeypatch.setattr(paired, "check_resource_headroom", lambda: None)
    TinySimulator.calls = 0
    config = ForwardScenarioConfig(paths=8, batch_size=4)
    arrays = {"loadings": np.zeros((24, 16, 2))}
    path = paired.generate_paired(arrays=arrays, metadata={"origin": "test"}, directory=tmp_path, split="train", config=config)
    assert TinySimulator.calls == 2
    with h5py.File(path, "r") as stream:
        assert stream.attrs["completed_paths"] == 8
        assert stream["diagnostics"].shape == (2, 20, 4)
        assert stream["core_features"].chunks is None
        assert stream["black0/payoff"].shape == (8,)
    first, _ = load_market_scenarios(path, split="train", case="black0", kind="cohorts")
    second, _ = load_market_scenarios(path, split="train", case="black0", kind="factors")
    np.testing.assert_array_equal(first.features[:], second.features[:])
    assert first.prices[:].shape == (8, 21, 169)
    assert second.prices[:].shape == (8, 21, 3)
    assert first.identity["model"]["simulated_background_volatility"] == 0
    paired.generate_paired(arrays=arrays, metadata={"origin": "test"}, directory=tmp_path, split="train", config=config)
    assert TinySimulator.calls == 2
    report = json.loads((tmp_path / "train_report.json").read_text())
    assert report["prices"]["black0"]["independent_pairs"] == 4
    (tmp_path / "train_report.json").unlink()
    paired.generate_paired(arrays=arrays, metadata={"origin": "test"}, directory=tmp_path, split="train", config=config)
    assert TinySimulator.calls == 2
    assert json.loads((tmp_path / "train_report.json").read_text())["complete"]


def test_partial_generation_resumes_from_last_complete_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(paired, "check_resource_headroom", lambda: None)
    original = TinySimulator.simulate
    seen = []

    class InterruptedSimulator(TinySimulator):
        def simulate(self, factors, background, progress):
            seen.append(factors.copy())
            if len(seen) == 2:
                raise InterruptedError("requested stop")
            return original(self, factors, background, progress)

    monkeypatch.setattr(paired, "PairedSimulator", InterruptedSimulator)
    options = dict(arrays={"loadings": np.zeros((24, 16, 2))}, metadata={"origin": "test"},
        directory=tmp_path, split="train", config=ForwardScenarioConfig(paths=8, batch_size=4))
    with pytest.raises(InterruptedError):
        paired.generate_paired(**options)
    with h5py.File(tmp_path / "train.h5", "r") as stream:
        assert stream.attrs["completed_paths"] == 4
    monkeypatch.setattr(paired, "PairedSimulator", TinySimulator)
    paired.generate_paired(**options)
    with h5py.File(tmp_path / "train.h5", "r") as stream:
        assert stream.attrs["completed_paths"] == 8
    assert json.loads((tmp_path / "train_report.json").read_text())["prices"]["black5"]["independent_pairs"] == 4


def test_incompatible_identity_cannot_overwrite_existing_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(paired, "PairedSimulator", TinySimulator)
    monkeypatch.setattr(paired, "check_resource_headroom", lambda: None)
    kwargs = dict(arrays={"loadings": np.zeros((24, 16, 2))}, directory=tmp_path,
        split="train", config=ForwardScenarioConfig(paths=4, batch_size=4))
    paired.generate_paired(**kwargs, metadata={"model": "old"})
    with pytest.raises(ValueError, match="identity changed"):
        paired.generate_paired(**kwargs, metadata={"model": "new"})


def test_pipeline_samples_latent_paths_once_then_trains_both_real_books(tmp_path):
    commands = stages(tmp_path, history=tmp_path / "h", estimate=tmp_path / "e", factor_law=tmp_path / "f",
        paths=1_000_000, batch_size=1024, maximum_epochs=2000)
    assert len(commands) == 5
    assert sum("quantdata.gromm2.paired_forward" in command for _, command in commands) == 1
    assert {name for name, _ in commands[1:]} == {"black0-factors", "black0-cohorts", "black5-factors", "black5-cohorts"}
