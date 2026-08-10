from __future__ import annotations

import numpy as np
import pytest

from quantdata.dysanos.grid.rndysanos.unconditional_option_test import (
    FLOAT32_UNIT_ROUNDOFF,
    SignAccumulator,
    delta_hedged_option_cost_notionals,
    delta_hedged_option_returns,
    delta_hedged_roundoff_band,
    net_return_roundoff_band,
    population_standard_error,
    rank_imbalance,
    rank_no_arbitrage_candidates,
)


def test_delta_hedged_option_returns_and_cost_notionals_exclude_spot() -> None:
    dC = np.array([[[[0.3, -0.2]]]], dtype=np.float32)
    delta = np.array([[[[0.5, -0.25]]]], dtype=np.float32)
    dS = np.array([[0.2]], dtype=np.float32)
    C = np.array([[[[1.0, 2.0]]]], dtype=np.float32)
    spot = np.array([[1.5]], dtype=np.float32)

    dH = delta_hedged_option_returns(dC=dC, delta=delta, dS=dS)
    H = delta_hedged_option_cost_notionals(C=C, delta=delta, spot=spot)

    assert dH.shape == H.shape == (1, 1, 2)
    assert dH[0, 0] == pytest.approx([0.2, -0.15])
    assert H[0, 0] == pytest.approx([1.75, 2.375])


def test_roundoff_bands_scale_with_operands_and_multiplier() -> None:
    C = np.array([[[[2.0]]]], dtype=np.float32)
    dC = np.array([[[[0.5]]]], dtype=np.float32)
    delta = np.array([[[[0.25]]]], dtype=np.float32)
    dS = np.array([[0.4]], dtype=np.float32)
    multiplier = 32.0

    actual = delta_hedged_roundoff_band(
        C=C,
        dC=dC,
        delta=delta,
        dS=dS,
        multiplier=multiplier,
    )
    scale = abs(2.5) + abs(2.0) + abs(0.1)
    assert actual.item() == pytest.approx(
        multiplier * FLOAT32_UNIT_ROUNDOFF * scale
    )

    values = np.array([[[0.2]]], dtype=np.float32)
    cost = np.array([[[0.01]]], dtype=np.float32)
    net_band = net_return_roundoff_band(values, cost, multiplier=multiplier)
    assert net_band.item() == pytest.approx(
        multiplier * FLOAT32_UNIT_ROUNDOFF * 0.21
    )


def test_sign_accumulator_distinguishes_strict_and_tolerance_signs() -> None:
    values = np.array(
        [
            [[-2.0e-4, 3.0], [-5.0e-5, -4.0]],
            [[2.0e-4, 2.0], [5.0e-5, -1.0]],
        ]
    )
    accumulator = SignAccumulator(2)
    accumulator.update(values[:1], 1.0e-4)
    accumulator.update(values[1:], 1.0e-4)
    result = accumulator.arrays()

    assert result["positive"].tolist() == [1, 2]
    assert result["negative"].tolist() == [1, 2]
    assert result["neutral"].tolist() == [2, 0]
    assert result["positive_paths"].tolist() == [1, 2]
    assert result["negative_paths"].tolist() == [1, 2]
    assert result["imbalance"].tolist() == [0, 0]
    assert result["observations"] == 4
    assert result["paths"] == 2
    assert result["steps"] == 2


def test_imbalance_ranking_uses_requested_count_and_labels() -> None:
    values = np.array(
        [
            [[1.0, -1.0, 1.0, 1.0], [1.0, -1.0, -1.0, 1.0]],
            [[1.0, -1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, 1.0]],
        ]
    )
    accumulator = SignAccumulator(4)
    accumulator.update(values, 0.0)

    ranked = rank_imbalance(accumulator, strikes=2, count=2)

    assert [item["option"] for item in ranked] == [1, 3]
    assert ranked[0]["expiry_index"] == 0
    assert ranked[0]["strike_index"] == 1
    assert ranked[1]["expiry_index"] == 1
    assert ranked[1]["strike_index"] == 1


def test_no_arbitrage_candidates_rank_zero_losses_first() -> None:
    long = SignAccumulator(2)
    short = SignAccumulator(2)
    long.update(np.array([[[1.0, 2.0]], [[1.0, -2.0]]]), 0.0)
    short.update(np.array([[[-1.0, -2.0]], [[-1.0, 2.0]]]), 0.0)

    ranked = rank_no_arbitrage_candidates(long, short, strikes=2, count=4)

    assert ranked[0]["option"] == 0
    assert ranked[0]["direction"] == "long"
    assert ranked[0]["negative"] == 0
    assert ranked[0]["mean"] == pytest.approx(1.0)


def test_population_standard_error_uses_all_observations() -> None:
    values = np.array([[1.0, -2.0], [3.0, 0.0], [5.0, 2.0], [7.0, 4.0]])
    actual = population_standard_error(
        values.sum(axis=0), np.square(values).sum(axis=0), len(values)
    )
    expected = values.std(axis=0, ddof=0) / np.sqrt(len(values))
    assert actual == pytest.approx(expected)
