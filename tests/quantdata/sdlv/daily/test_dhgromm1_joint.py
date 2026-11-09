"""The sum objective updates separate bid/ask networks without gradient sharing."""

import copy
import csv
import json
from dataclasses import asdict

import pytest
import torch

from quantdata.sdlv.dhgromm1.feedback import FeedbackConfig
from quantdata.sdlv.dhgromm1.hedge import oce_value
from quantdata.sdlv.dhgromm1.joint import (
    JointConfig,
    JointHedges,
    load_joint_checkpoint,
    load_joint_model,
)
from quantdata.sdlv.dhgromm1.plot_losses import joint_history, plot_joint_losses
from quantdata.torch import BaseEnvironment


def models_and_data():
    torch.manual_seed(78)
    env = BaseEnvironment(device="cpu", precision=64, debug=False)
    config = JointConfig(label="test", long=FeedbackConfig(label="long", hidden_size=3, latent_width=4),
                         short=FeedbackConfig(label="short", hidden_size=3, latent_width=4))
    model = JointHedges(env, config=config, means={k: torch.zeros(21) for k in ("long", "short")},
                        scales={k: torch.ones(21) for k in ("long", "short")})
    with torch.no_grad():
        for policy in (model.long, model.short):
            policy.trade_head.weight.add_(torch.randn_like(policy.trade_head.weight) * .001)
            policy.deep_head.I0.weight.add_(torch.randn_like(policy.deep_head.I0.weight) * .001)
    price = .8 + torch.rand(4, 21, 21, dtype=torch.float64)
    state = .1 * torch.randn(4, 20, 4, dtype=torch.float64)
    ret = .01 * torch.randn(4, 20, 21, dtype=torch.float64)
    payoff = torch.tensor([.02, .01, .025, .015], dtype=torch.float64)
    return model, (price, state, ret, payoff)


def test_sum_backward_and_adam_equal_independent_network_updates() -> None:
    joint, inputs = models_and_data()
    independent = copy.deepcopy(joint)
    joint_optimizer = torch.optim.Adam([dict(params=joint.long.parameters()),
                                        dict(params=joint.short.parameters())], lr=1e-5)
    joint_values = joint(*inputs, eager=True)
    joint_values.sum().backward()
    price, state, ret, payoff = inputs
    expected_losses = []
    for label, sign in (("long", 1), ("short", -1)):
        policy = getattr(independent, label)
        result = policy(price, state, returns=ret, eager=True)
        value = -oce_value(result.gain + sign * payoff, policy.translation, 100) + result.penalty.mean()
        expected_losses.append(value.detach())
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-5)
        value.backward()
        optimizer.step()
    torch.testing.assert_close(joint_values.detach(), torch.stack(expected_losses), rtol=0, atol=0)
    assert joint.long.translation.grad.abs() > 0
    assert joint.short.translation.grad.abs() > 0
    joint_optimizer.step()
    for name, value in joint.state_dict().items():
        torch.testing.assert_close(value, independent.state_dict()[name], rtol=1e-12, atol=1e-12, msg=name)


def test_networks_and_translations_have_independent_storage() -> None:
    model, _ = models_and_data()
    assert model.long.translation.data_ptr() != model.short.translation.data_ptr()
    assert not ({p.data_ptr() for p in model.long.parameters()} & {p.data_ptr() for p in model.short.parameters()})


def test_scans_preserve_nonuniform_oce_gradient_weights() -> None:
    model, inputs = models_and_data()
    reference = copy.deepcopy(model)
    model(*inputs).sum().backward()
    reference(*inputs, eager=True).sum().backward()
    for (name, p), (_, q) in zip(model.named_parameters(), reference.named_parameters(), strict=True):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-9, atol=1e-11, msg=name)


def test_incompatible_joint_checkpoint_rejected_before_model_build(tmp_path) -> None:
    path = tmp_path / "wrong.pt"
    torch.save(dict(model_code_version="old-sequential-objective"), path)
    with pytest.raises(ValueError, match="objective, version or geometry differs"):
        load_joint_checkpoint(path, config=JointConfig(label="bid-ask"))


def test_selected_pair_can_be_loaded_without_training(tmp_path) -> None:
    model, inputs = models_and_data()
    path = tmp_path / "joint.pt"
    torch.save(dict(model_config=asdict(model.config), model_code_version=JointConfig.version.unique_id48,
        user_version=model.config.user_version, best_state=model.state_dict()), path)
    loaded = load_joint_model(path, config=model.config, env=BaseEnvironment(device="cpu", precision=64))
    torch.testing.assert_close(loaded(*inputs, eager=True), model(*inputs, eager=True))
    assert not any(p.requires_grad for p in loaded.parameters())


def test_joint_loss_plot_selects_a_common_training_sum_epoch(tmp_path) -> None:
    run, output = tmp_path / "run", tmp_path / "figure"
    run.mkdir()
    (run / "identity.json").write_text(json.dumps(dict(joint_epochs=500)))
    with (run / "joint-losses.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("epoch", "long_loss", "short_loss", "sum_loss", "online_sum_loss"))
        writer.writerows(((0, -.02, .03, .01, .01), (1, -.03, .05, .02, .03), (2, -.01, .015, .005, .02)))
    summary = plot_joint_losses(run, output, partial=True)
    assert summary["selected_epoch"] == 2  # Not the long policy's own minimum.
    assert summary["selected"]["long_loss"] == -.01
    assert summary["validation"] is False
    assert (output / "losses.png").is_file()
    assert len(joint_history(run)) == 3


def test_observable_joint_plot_labels_log_axes_and_rejects_negative_values(tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "identity.json").write_text(json.dumps(dict(joint_epochs=2, excluded_inputs=["simulator latent state"])))
    path = run / "joint-losses.csv"
    path.write_text("epoch,long_loss,short_loss,sum_loss,online_sum_loss\n0,1,2,3,3\n1,.1,.2,.3,1\n2,5,6,11,4\n")
    result = plot_joint_losses(run, tmp_path / "log-figure", partial=True, log_y=True)
    assert result["policy_inputs"] == "observable-only"
    assert result["y_scale"] == "log"
    assert result["selected_epoch"] == 1
    assert result["validation"] is False
    path.write_text("epoch,long_loss,short_loss,sum_loss,online_sum_loss\n0,-1,2,1,1\n")
    with pytest.raises(ValueError, match="strictly positive"):
        plot_joint_losses(run, tmp_path / "invalid", partial=True, log_y=True)


@pytest.mark.parametrize("row", ["2,0,1,1,1", "1,0,1,2,1", "1,nan,1,nan,1"])
def test_joint_history_rejects_rewound_invalid_or_nonfinite_losses(tmp_path, row: str) -> None:
    (tmp_path / "joint-losses.csv").write_text(
        "epoch,long_loss,short_loss,sum_loss,online_sum_loss\n0,-.02,.03,.01,.01\n" + row + "\n")
    with pytest.raises(ValueError):
        joint_history(tmp_path)
