"""Signed loss scheduling, plateau semantics and exact schedule restarts."""

import copy
import math

import pytest
import torch

from quantdata.hedge2.terminal_schedule import step_training_scheduler


def make_scheduler(*, patience: int = 2, cooldown: int = 0, minimum: float = .01):
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.tensor(1.))], lr=1.)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5,
        patience=patience, cooldown=cooldown, min_lr=minimum)
    return optimizer, scheduler


@pytest.mark.parametrize("best,worse", ((.1, .10001), (-.1, -.09999), (-1., -.95), (0., .01)))
def test_worsening_is_never_counted_as_improvement(best: float, worse: float) -> None:
    _, scheduler = make_scheduler()
    step_training_scheduler(scheduler, best, tolerance=.1)
    step_training_scheduler(scheduler, worse, tolerance=.1)
    assert scheduler.best == best
    assert scheduler.num_bad_epochs == 1
    assert scheduler.threshold_mode == "abs"
    assert scheduler.threshold == .1


def test_subunit_threshold_has_a_unit_floor_and_exact_boundary_is_not_improvement() -> None:
    _, scheduler = make_scheduler()
    step_training_scheduler(scheduler, .5, tolerance=.125)
    step_training_scheduler(scheduler, .375, tolerance=.125)
    assert scheduler.best == .5
    assert scheduler.num_bad_epochs == 1
    step_training_scheduler(scheduler, .25, tolerance=.125)
    assert scheduler.best == .25
    assert scheduler.num_bad_epochs == 0


def test_above_unit_threshold_uses_previous_best_not_current_loss() -> None:
    _, scheduler = make_scheduler()
    step_training_scheduler(scheduler, 10., tolerance=.125)
    step_training_scheduler(scheduler, 8.75, tolerance=.125)
    assert scheduler.best == 10.
    assert scheduler.threshold == 1.25
    step_training_scheduler(scheduler, 8.5, tolerance=.125)
    assert scheduler.best == 8.5
    assert scheduler.threshold == 1.25
    step_training_scheduler(scheduler, 8.4, tolerance=.125)
    assert scheduler.threshold == 1.0625


def test_cumulative_small_improvements_are_compared_to_running_best() -> None:
    _, scheduler = make_scheduler()
    for loss in (1., .96, .92):
        step_training_scheduler(scheduler, loss, tolerance=.05)
    assert scheduler.best == .92
    assert scheduler.num_bad_epochs == 0


def test_patience_cooldown_and_learning_rate_floor_remain_torch_semantics() -> None:
    optimizer, scheduler = make_scheduler(patience=1, cooldown=2, minimum=.25)
    rates = []
    for _ in range(11):
        step_training_scheduler(scheduler, -.1, tolerance=1e-6)
        rates.append(optimizer.param_groups[0]["lr"])
    assert rates == [1., 1., .5, .5, .5, .5, .25, .25, .25, .25, .25]


def test_save_resume_preserves_scheduler_and_optimizer_sequence_exactly() -> None:
    losses = [.12, .12, .1199999, -.02, -.0199999, -.02, -.02001, -.02001, -.02001]
    full_optimizer, full = make_scheduler(patience=1, cooldown=1)
    split_optimizer, split = make_scheduler(patience=1, cooldown=1)
    for index, loss in enumerate(losses):
        step_training_scheduler(full, loss, tolerance=1e-6)
        if index < 4:
            step_training_scheduler(split, loss, tolerance=1e-6)
        if index == 3:
            optimizer_state, schedule_state = copy.deepcopy(split_optimizer.state_dict()), copy.deepcopy(split.state_dict())
            resumed_optimizer, resumed = make_scheduler(patience=1, cooldown=1)
            resumed_optimizer.load_state_dict(optimizer_state)
            resumed.load_state_dict(schedule_state)
        if index >= 4:
            step_training_scheduler(resumed, loss, tolerance=1e-6)
            assert resumed.state_dict() == full.state_dict()
            assert resumed_optimizer.state_dict() == full_optimizer.state_dict()


@pytest.mark.parametrize("loss,tolerance", ((math.nan, .1), (math.inf, .1), (.1, math.nan), (.1, -.1)))
def test_invalid_numbers_do_not_mutate_the_scheduler(loss: float, tolerance: float) -> None:
    _, scheduler = make_scheduler()
    before = copy.deepcopy(scheduler.state_dict())
    with pytest.raises(ValueError, match="finite"):
        step_training_scheduler(scheduler, loss, tolerance=tolerance)
    assert scheduler.state_dict() == before
