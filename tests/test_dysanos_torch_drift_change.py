import sys
from types import SimpleNamespace

import numpy as np
import torch

# The module's default model configuration derives its label from a notebook.
sys.modules["ipynbname"] = SimpleNamespace(name=lambda: "test")

from quantdata.dysanos.grid.torch_drift_changeV2_1 import (
    _load_rng_state_dict,
    _rng_state_dict,
)
from quantdata.dysanos.grid.torch_drift_changeV2 import DRN, DRNConfig


def test_rng_state_roundtrip_and_old_checkpoint_compatibility():
    np.random.seed(17)
    torch.manual_seed(19)
    state = _rng_state_dict()

    assert "python" not in state
    expected_numpy = np.random.random()
    expected_torch = torch.rand(4)

    np.random.seed(103)
    torch.manual_seed(107)
    np.random.random()
    torch.rand(4)

    _load_rng_state_dict(state)
    assert np.random.random() == expected_numpy
    assert torch.equal(torch.rand(4), expected_torch)

    # Checkpoints written before RNG serialization remain loadable.
    _load_rng_state_dict(None)
    _load_rng_state_dict({})


def test_random_bootstrap_axis_draws_fresh_multinomial_counts():
    torch.manual_seed(23)
    first = DRN._random_bootstrap_axis(
        31, 7, dtype=torch.float32, device=torch.device("cpu"))
    second = DRN._random_bootstrap_axis(
        31, 7, dtype=torch.float32, device=torch.device("cpu"))

    assert first.shape == second.shape == (31, 7)
    assert torch.equal(first.sum(0), torch.full((7,), 31.0))
    assert torch.equal(second.sum(0), torch.full((7,), 31.0))
    assert torch.all(first >= 0) and torch.all(first == first.round())
    assert not torch.equal(first, second)


def test_boot_random_defaults_off_and_can_be_enabled():
    assert DRNConfig(label="test").boot_random is False
    assert DRNConfig(label="test", boot_random=True).boot_random is True
