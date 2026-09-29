from __future__ import annotations

import torch

from quantdata.dysanos.flex.network_oce import (
    conditional_entropic_oce,
    entropic_density_from_pnl,
    l1_capped_action,
)


def test_l1_action_cap_preserves_interior_and_caps_exterior() -> None:
    raw = torch.tensor(((1.0, -2.0), (6.0, -2.0)), dtype=torch.float64)
    bounded = l1_capped_action(raw, 4.0)
    torch.testing.assert_close(bounded[0], raw[0])
    torch.testing.assert_close(bounded[1], torch.tensor((3.0, -1.0), dtype=raw.dtype))


def test_transaction_cost_creates_the_correct_no_trade_band() -> None:
    returns = torch.tensor(
        (((-0.01,), (0.03,), (-0.01,), (0.03,)),), dtype=torch.float64
    )
    action = torch.zeros((1, 1), dtype=torch.float64, requires_grad=True)
    cost = torch.tensor((0.025,), dtype=torch.float64)

    values = conditional_entropic_oce(returns, action, cost)
    values.loss.backward()

    # The unweighted drift is 1%, inside the 2.5% proportional-cost band, so
    # zero belongs to the nonsmooth objective's subgradient interval.
    assert abs(float(values.weighted_return_mean[0, 0].detach())) < float(cost[0])
    torch.testing.assert_close(torch.mean(values.weights, dim=1), torch.ones(1, dtype=cost.dtype))


def test_replayed_entropic_density_has_exact_log_oce_gradient() -> None:
    returns = torch.tensor((-0.2, 0.1, 0.7, -0.4), dtype=torch.float64)
    direct_action = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    direct_pnl = direct_action * returns
    direct = entropic_density_from_pnl(direct_pnl)
    direct.log_oce.backward()

    replay_action = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    replay_pnl = replay_action * returns
    replay = entropic_density_from_pnl(replay_pnl.detach())
    surrogate = -torch.mean(replay.weights.detach() * replay_pnl)
    surrogate.backward()

    torch.testing.assert_close(torch.mean(replay.weights), torch.ones((), dtype=torch.float64))
    torch.testing.assert_close(replay_action.grad, direct_action.grad)
