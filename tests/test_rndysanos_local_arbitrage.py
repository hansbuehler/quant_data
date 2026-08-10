from __future__ import annotations

import numpy as np
import torch

from quantdata.dysanos.grid.rndysanos.local_arbitrage import (
    LocalArbitrageConfig,
    LocalArbitrageProgress,
    cost_adjusted_direction_candidates,
    pack_direction_mask,
    unpack_direction_mask,
    valuate_local_arbitrage_mask,
)


def test_cost_adjusted_candidates_subtract_cost_in_both_directions() -> None:
    returns = torch.tensor(
        [[[0.4, -0.4], [0.3, -0.3], [0.2, -0.2]]]
    )
    cost = torch.tensor([[0.25, 0.1]])
    roundoff = torch.full_like(returns, 0.01)

    candidates = cost_adjusted_direction_candidates(returns, cost, roundoff)

    assert torch.equal(candidates, torch.tensor([[[False, False], [False, True]]]))


def test_local_arbitrage_mask_streams_training_fitted_candidates() -> None:
    config = LocalArbitrageConfig(
        neighbors=2, query_batch_size=1, instrument_batch_size=2
    )
    reference_state = torch.tensor([[0.0], [1.0], [10.0], [11.0]])
    reference_returns = torch.tensor(
        [[0.3, 0.1], [0.2, -0.1], [0.1, -0.2], [-0.1, -0.3]]
    )
    cost = torch.full_like(reference_returns, 0.05)
    roundoff = torch.full_like(reference_returns, 0.01)
    streamed: list[np.ndarray] = []
    progress: list[LocalArbitrageProgress] = []

    report = valuate_local_arbitrage_mask(
        reference_state=reference_state,
        reference_returns=reference_returns,
        reference_roundoff=roundoff,
        query_state=torch.tensor([[0.1], [10.1]]),
        query_returns=torch.tensor([[0.25, 0.0], [0.0, -0.25]]),
        query_cost=torch.full((2, 2), 0.05),
        query_roundoff=torch.full((2, 2), 0.01),
        config=config,
        split="validation",
        leave_one_out=False,
        candidate_sink=lambda start, stop, value: streamed.append(value.copy()),
        progress=progress.append,
        progress_interval_seconds=1.0e9,
    )

    mask = np.concatenate(streamed, axis=0)
    assert mask.shape == (2, 2, 2)
    assert np.array_equal(mask[:, 0].any(axis=1), np.array([True, False]))
    assert np.array_equal(mask[:, 1].any(axis=1), np.array([False, True]))
    assert progress[-1].long_only_candidate_states == 1
    assert progress[-1].short_only_candidate_states == 1
    assert report["states_by_direction_exclusivity"] == {
        "long_only": 1,
        "short_only": 1,
        "both_directions": 0,
        "neither_direction": 0,
    }


def test_direction_mask_pack_round_trip_on_torch_device() -> None:
    rng = np.random.default_rng(913)
    mask = rng.random((3, 4, 2, 13)) < 0.3

    packed = pack_direction_mask(mask)
    unpacked = unpack_direction_mask(torch.from_numpy(packed), instruments=13)

    assert packed.shape == (3, 4, 2, 2)
    assert np.array_equal(unpacked.numpy(), mask)
