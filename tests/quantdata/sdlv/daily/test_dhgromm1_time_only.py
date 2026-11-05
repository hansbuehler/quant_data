"""Exact deterministic actions, inside-utility costs, and compressed P&L gradients."""

import copy
from dataclasses import asdict

import numpy as np
import pytest
import torch

from quantdata.sdlv.dhgromm1.time_only import (
    LABELS,
    SIGNS,
    TimeOnlyConfig,
    TimeOnlyHedge,
    TimeOnlyTriple,
    return_exposures,
)
from quantdata.sdlv.dhgromm1.time_only_run import load_time_only_checkpoint, load_time_only_model
from quantdata.torch import BaseEnvironment


def model() -> TimeOnlyTriple:
    return TimeOnlyTriple(BaseEnvironment(device="cpu", precision=64), config=TimeOnlyConfig(label="test"))


def test_actions_are_one_affine_vector_plus_two_exact_date_vectors() -> None:
    m = model()
    assert sum(p.numel() for p in m.parameters()) == 3 * (4 * 21 + 1)
    with torch.no_grad():
        p = m.long
        p.affine.I0.bias.fill_(1)
        p.affine.I0.weight.fill_(2)
        p.day0.fill_(3)
        p.day10.fill_(4)
    expected = (1 + 2 * torch.arange(20, dtype=torch.float64) / 20)[:, None].expand(-1, 21).clone()
    expected[0] += 3
    expected[10] += 4
    torch.testing.assert_close(p.actions(), expected, rtol=0, atol=0)
    assert not any(isinstance(c, (torch.nn.LSTM, torch.nn.LSTMCell)) for c in m.modules())


def test_initial_actions_are_zero_and_independent_of_random_seed() -> None:
    torch.manual_seed(1)
    a = model()
    torch.manual_seed(92)
    b = model()
    for key, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)
    for label in LABELS:
        assert torch.count_nonzero(getattr(a, label).actions()) == 0


def test_cost_is_applied_to_combined_actions_and_is_INSIDE_utility() -> None:
    m = model()
    with torch.no_grad():
        m.empty.affine.I0.bias.fill_(.1)
        m.empty.day0.fill_(-.1)  # Combined action cancels on day 0.
    actions = m.empty.actions()
    assert torch.count_nonzero(actions[0]) == 0
    c = m.empty.cost()
    expected_cost = 19 * 21 * (1e-4 * .1 + 1e-8 * .1**2)
    assert c.item() == pytest.approx(expected_cost)
    losses = m(torch.zeros(4, 84, dtype=torch.float64), torch.zeros(4, dtype=torch.float64))
    # At translation zero, -u(-C)=C+lambda*C^2/2, NOT an external +C.
    assert losses[0].item() == pytest.approx(expected_cost + 50 * expected_cost**2)
    assert losses[0].item() > expected_cost
    losses.sum().backward()
    assert m.empty.translation.grad.item() == pytest.approx(100 * expected_cost)


def test_compressed_gain_matches_raw_return_sum_and_every_gradient() -> None:
    m = model()
    torch.manual_seed(9)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(.01 * torch.randn_like(p))
    ref = copy.deepcopy(m)
    rng = np.random.default_rng(5)
    ret = rng.normal(0, .01, (8, 20, 21))
    x = torch.as_tensor(return_exposures(ret, m.config))
    h = torch.linspace(.01, .04, 8, dtype=torch.float64)
    actual = m(x, h)
    direct = []
    for label, sign in zip(LABELS, SIGNS, strict=True):
        policy = getattr(ref, label)
        gain = (torch.as_tensor(ret) * policy.actions()).sum((1, 2))
        net = gain + sign * h - policy.cost() - policy.translation
        direct.append(-(policy.translation + ref._utility(net, torch.zeros_like(net)).mean()))
    expected = torch.stack(direct)
    torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-12)
    actual.sum().backward()
    expected.sum().backward()
    for (name, p), (_, q) in zip(m.named_parameters(), ref.named_parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-12, rtol=1e-10, msg=name)


def test_zero_affine_initialization_still_has_action_parameter_gradients() -> None:
    m = model()
    x = torch.ones(4, 84, dtype=torch.float64) * .01
    m(x, torch.ones(4, dtype=torch.float64)*.02).sum().backward()
    for label in LABELS:
        policy = getattr(m, label)
        assert policy.affine.I0.weight.grad.abs().sum() > 0
        assert policy.affine.I0.bias.grad.abs().sum() > 0
        assert policy.day0.grad.abs().sum() > 0
        assert policy.day10.grad.abs().sum() > 0


def test_all_parameter_types_match_independent_numpy_finite_differences() -> None:
    m = model()
    rng = np.random.default_rng(61)
    ret = rng.normal(0, .03, (12, 20, 21))
    x = return_exposures(ret, m.config)
    payoff = rng.uniform(.01, .04, 12)
    with torch.no_grad():
        for p in m.parameters():
            p.fill_(.003)
    m(torch.as_tensor(x), torch.as_tensor(payoff)).sum().backward()

    def numpy_loss(policy, sign):
        # Independent value calculation: raw returns, manually defined
        # time schedule, total cost and the explicit utility formula.
        b = policy.affine.I0.bias.detach().numpy()
        w = policy.affine.I0.weight.detach().numpy()[:, 0]
        actions = b[None] + np.arange(20)[:, None] / 20 * w[None]
        actions = actions.copy()
        actions[0] += policy.day0.detach().numpy()
        actions[10] += policy.day10.detach().numpy()
        cost = np.sum(1e-4 * np.abs(actions) + 1e-8 * actions**2)
        cash = float(policy.translation.detach())
        z = sign * payoff + np.sum(ret * actions, axis=(1, 2)) - cost - cash
        utility = np.where(z >= 0, -np.expm1(-100 * np.maximum(z, 0)) / 100,
                           np.minimum(z, 0) - 50 * np.minimum(z, 0)**2)
        return -cash - utility.mean()

    for label, sign in zip(LABELS, SIGNS, strict=True):
        policy = getattr(m, label)
        for p in policy.parameters():
            epsilon = 1e-6
            gradient = float(p.grad.flatten()[0])
            with torch.no_grad():
                flat = p.view(-1)
                original = float(flat[0])
                flat[0] = original + epsilon
                plus = numpy_loss(policy, sign)
                flat[0] = original - epsilon
                minus = numpy_loss(policy, sign)
                flat[0] = original
            assert gradient == pytest.approx((plus - minus) / (2 * epsilon), abs=1e-8, rel=1e-5)


def test_valuation_load_never_trains_and_rejects_old_cost_placement(tmp_path) -> None:
    m = model()
    path = tmp_path / "model.pt"
    payload = dict(model_config=asdict(m.config), model_code_version=TimeOnlyConfig.version.unique_id48,
        user_version=m.config.user_version, best_state=m.state_dict(), cost_treatment="inside_utility")
    torch.save(payload, path)
    loaded = load_time_only_model(path, config=m.config, env=BaseEnvironment(device="cpu", precision=64))
    assert not any(p.requires_grad for p in loaded.parameters())
    payload["cost_treatment"] = "outside_utility"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="cost placement"):
        load_time_only_checkpoint(path, config=m.config)
