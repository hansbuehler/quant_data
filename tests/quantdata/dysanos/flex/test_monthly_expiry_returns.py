from __future__ import annotations

from types import SimpleNamespace

import torch

from quantdata.dysanos.flex.run_single_state_monthly_network_oce import (
    _book,
    _expiry_option_returns,
    _nms_book,
    _nms_relative_strikes,
)


def test_expiry_returns_use_fixed_cash_strikes_and_terminal_intrinsic_value() -> None:
    returns = _expiry_option_returns(
        gross_horizon=torch.tensor((0.90, 1.10), dtype=torch.float64),
        issue_values=torch.tensor(((0.10, 0.20), (0.10, 0.20)), dtype=torch.float64),
        strikes=torch.tensor((0.99, 1.01), dtype=torch.float64),
        is_put=torch.tensor((False, True)),
    )

    expected = torch.tensor(((-1.0, -0.45), (0.10, -1.0)), dtype=torch.float64)
    assert torch.allclose(returns, expected)


def test_expanded_floating_book_has_three_options_per_expiry() -> None:
    book = _book((5, 10, 15, 20))

    assert len(book.labels) == 13
    assert len(book.relative_strikes) == 12
    assert book.initial_days == (
        5.0,
        5.0,
        5.0,
        10.0,
        10.0,
        10.0,
        15.0,
        15.0,
        15.0,
        20.0,
        20.0,
        20.0,
    )
    assert book.final_days[-3:] == (19.0, 19.0, 19.0)


def test_half_range_nms_book_uses_eleven_itm_options_per_expiry() -> None:
    scores = tuple(-1.5 + 0.2 * index for index in range(11))
    book = _nms_book((5, 10, 15, 20), normalized_moneyness=scores)

    assert len(book.labels) == 45
    assert book.relative_strikes is None
    assert book.normalized_moneyness is not None
    assert book.normalized_moneyness[:11] == scores
    assert book.is_put[:11] == (False,) * 8 + (True,) * 3
    assert book.labels[8] == "5D NMS -0.1 ITM call"


def test_nms_strikes_use_each_paths_current_total_volatility() -> None:
    nodes = torch.tensor((5.0, 10.0), dtype=torch.float64) / 252.0
    model = SimpleNamespace(
        sanos_decoder=SimpleNamespace(model_expiries_y=nodes)
    )
    surface = SimpleNamespace(
        total_vols=torch.tensor(
            ((0.10, 0.20), (0.20, 0.40)), dtype=torch.float64
        )
    )
    days = torch.tensor((5.0, 10.0), dtype=torch.float64)
    scores = torch.tensor((-1.0, 0.5), dtype=torch.float64)

    strikes = _nms_relative_strikes(
        model,
        surface,
        days=days,
        normalized_moneyness=scores,
    )

    assert strikes.shape == (2, 2)
    assert torch.all(strikes[:, 0] < 1.0)
    assert torch.all(strikes[:, 1] > 1.0)
    assert strikes[1, 0] < strikes[0, 0]
    assert strikes[1, 1] > strikes[0, 1]


def test_expiry_returns_accept_pathwise_issue_strikes() -> None:
    returns = _expiry_option_returns(
        gross_horizon=torch.tensor((0.90, 1.10), dtype=torch.float64),
        issue_values=torch.tensor(((0.10,), (0.10,)), dtype=torch.float64),
        strikes=torch.tensor(((0.99,), (1.01,)), dtype=torch.float64),
        is_put=torch.tensor((False,)),
    )

    assert torch.allclose(
        returns,
        torch.tensor(((-1.0,), (-0.10,)), dtype=torch.float64),
    )
