from __future__ import annotations

from pathlib import Path

import pytest

from quantdata.gromm2 import market_hedge_schedule as schedule
from quantdata.gromm2.joint_hedging import JointTrainConfig
from quantdata.gromm2.market_hedge_schedule import hedge_commands


def test_scheduled_books_have_separate_outputs_and_same_controls():
    commands = hedge_commands(Path("/source"), Path("/output"), maximum_epochs=2000, memory_fraction=1.0)
    assert set(commands) == {"cohorts", "factors"}
    for kind, command in commands.items():
        assert command[command.index("--output-directory") + 1] == f"/output/policies/{kind}"
        assert command[command.index("--spot-cost-bps") + 1] == "0.1"
        assert command[command.index("--cuda-memory-fraction") + 1] == "1.0"
        assert command[command.index("--maximum-epochs") + 1] == "2000"
        assert ("--with-spot" in command) == (kind == "cohorts")


@pytest.mark.parametrize("fraction", [0, -0.1, 1.1, float("nan")])
def test_unsafe_memory_configuration_is_rejected(fraction):
    with pytest.raises(ValueError, match="memory fraction"):
        hedge_commands(Path("/source"), Path("/output"), maximum_epochs=2000, memory_fraction=fraction)


@pytest.mark.parametrize("fraction", [0, -0.1, 1.1, float("nan")])
def test_training_allocator_limit_is_validated(fraction):
    with pytest.raises(ValueError, match="cuda_memory_fraction"):
        JointTrainConfig(cuda_memory_fraction=fraction)


def test_second_book_is_not_launched_until_first_has_exited(tmp_path, monkeypatch):
    children = []
    monkeypatch.setattr(schedule, "_require_cuda_visibility", lambda: None)
    monkeypatch.setattr(schedule, "check_resource_headroom", lambda: None)

    class CompletedChild:
        def __init__(self, command, **kwargs):
            assert not children or children[-1].returncode == 0
            self.command, self.returncode, self.pid = command, None, 100 + len(children)
            self.poll_count = 0
            children.append(self)

        def poll(self):
            self.poll_count += 1
            if self.poll_count > 1:
                self.returncode = 0
            return self.returncode

    monkeypatch.setattr(schedule.subprocess, "Popen", CompletedChild)
    assert schedule.run_schedule(tmp_path / "source", tmp_path / "outputs") == 0
    assert [child.command[child.command.index("--kind") + 1] for child in children] == ["cohorts", "factors"]
