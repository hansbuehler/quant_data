"""Checkpoint-lineage gates for consecutive-date VolGAN hedges."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

from quantdata.sdlv.daily.front_spot.hedging.models.checkpointing import file_sha256
from quantdata.sdlv.daily.front_spot.hedging.models.volgan.data import (
    has_exact_one_day_target,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.volgan_rolling_chain import (
    VolganRollingChain,
)
from quantdata.torch import torch


def test_volgan_rolling_chain_requires_previous_complete_fit(tmp_path) -> None:
    initial = tmp_path / "initial.pt"
    torch.save(
        {"training_end_date": "2022-12-30", "dataset_manifest_sha256": "a" * 64},
        initial,
    )
    checkpoint_directory = tmp_path / "rolling"
    checkpoint_directory.mkdir()
    chain = VolganRollingChain(
        initial_checkpoint=initial,
        initial_sha256=file_sha256(initial),
        checkpoint_directory=checkpoint_directory,
        dataset_manifest_sha256="a" * 64,
    )
    first_date = date(2024, 4, 1)
    assert chain.source(first_date)[0] == initial
    first = checkpoint_directory / "first.pt"
    torch.save(
        {
            "training_end_date": first_date.isoformat(),
            "warm_start_source_sha256": file_sha256(initial),
            "dataset_manifest_sha256": "a" * 64,
            "epoch": 9,
            "train_config": {"epochs": 10},
        },
        first,
    )
    first_result = SimpleNamespace(
        diagnostics={
            "warm_start_sha256": file_sha256(initial),
            "fit_checkpoint_sha256": file_sha256(first),
            "fit_checkpoint_path": str(first),
        }
    )
    chain.advance(first_date, (first_result,))
    assert chain.carry_gap(date(2024, 4, 2)) == file_sha256(first)
    assert chain.source(date(2024, 4, 3))[0] == first
    with pytest.raises(ValueError, match="lineage"):
        chain.advance(
            date(2024, 4, 3),
            (SimpleNamespace(diagnostics={**first_result.diagnostics, "warm_start_sha256": "b" * 64}),),
        )
    with pytest.raises(ValueError, match="missing or changed"):
        chain.advance(
            date(2024, 4, 3),
            (
                SimpleNamespace(
                    diagnostics={
                        "warm_start_sha256": file_sha256(first),
                        "fit_checkpoint_sha256": "c" * 64,
                        "fit_checkpoint_path": str(first),
                    }
                ),
            ),
        )


def test_volgan_exact_target_requires_three_one_session_lags() -> None:
    dates = tuple(date(2024, 4, day) for day in range(1, 7))

    class Observations:
        def row(self, index, *, named):
            assert named
            return {
                "business_sessions_from_previous_observation": (1, 1, 2, 1, 1, 1)[index]
            }

    panel = SimpleNamespace(dates=dates, observations=Observations())
    assert has_exact_one_day_target(panel, dates[2]) is False
    assert has_exact_one_day_target(panel, dates[3]) is False
    assert has_exact_one_day_target(panel, dates[4]) is False
    assert has_exact_one_day_target(panel, dates[5]) is True
