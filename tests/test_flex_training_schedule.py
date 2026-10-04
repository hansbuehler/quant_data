from __future__ import annotations

import pytest
import torch

from quantdata.dysanos.flex.config import FlexTrainConfig
from quantdata.dysanos.flex.training import _learning_rate_multiplier


def test_cosine_schedule_reaches_floor_without_rebound() -> None:
    config = FlexTrainConfig(
        num_epochs=10,
        learning_rate=1.0e-3,
        minimum_learning_rate_fraction=0.1,
    )
    multipliers = [_learning_rate_multiplier(epoch, config) for epoch in range(12)]
    assert multipliers[0] == pytest.approx(1.0)
    assert multipliers[9] == pytest.approx(0.1)
    assert multipliers[10] == pytest.approx(0.1)
    assert multipliers[11] == pytest.approx(0.1)
    assert all(
        left >= right for left, right in zip(multipliers, multipliers[1:])
    )


def test_warmup_cosine_schedule_reaches_peak_then_floor() -> None:
    config = FlexTrainConfig(
        num_epochs=10,
        learning_rate=1.0e-3,
        learning_rate_warmup_epochs=3,
        minimum_learning_rate_fraction=0.1,
    )
    multipliers = [_learning_rate_multiplier(epoch, config) for epoch in range(10)]
    assert multipliers[:4] == pytest.approx([0.4, 0.7, 1.0, 1.0])
    assert multipliers[-1] == pytest.approx(0.1)


def test_lambda_scheduler_uses_the_expected_rate_each_epoch() -> None:
    config = FlexTrainConfig(
        num_epochs=6,
        learning_rate=2.0e-3,
        learning_rate_warmup_epochs=2,
        minimum_learning_rate_fraction=0.25,
    )
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.Adam([parameter], lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: _learning_rate_multiplier(epoch, config),
    )
    used_rates: list[float] = []
    for _ in range(config.num_epochs):
        used_rates.append(float(optimizer.param_groups[0]["lr"]))
        optimizer.step()
        scheduler.step()
    expected = [
        config.learning_rate * _learning_rate_multiplier(epoch, config)
        for epoch in range(config.num_epochs)
    ]
    assert used_rates == pytest.approx(expected)
    assert used_rates[-1] == pytest.approx(
        config.learning_rate * config.minimum_learning_rate_fraction
    )


def test_decoder_learning_rate_can_be_scaled_separately() -> None:
    config = FlexTrainConfig(
        learning_rate=2.0e-2,
        decoder_learning_rate=2.0e-4,
    )

    assert config.learning_rate == pytest.approx(2.0e-2)
    assert config.decoder_learning_rate == pytest.approx(2.0e-4)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"learning_rate_schedule": "bad"},
        {"num_epochs": 5, "learning_rate_warmup_epochs": 5},
        {"minimum_learning_rate_fraction": -0.1},
        {"minimum_learning_rate_fraction": 1.1},
        {"decoder_learning_rate": 0.0},
    ],
)
def test_invalid_learning_rate_schedule_is_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        FlexTrainConfig(**kwargs)  # type: ignore[arg-type]
