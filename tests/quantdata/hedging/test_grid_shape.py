"""Block factor separation, joint spot accounting and grid/shape checkpoint replay."""

from pathlib import Path

import numpy as np
import pytest

from quantdata.hedge2.dlv import DlvConfig
from quantdata.hedge2.dynamics import DynamicsConfig
from quantdata.hedging.model_audit.grid_shape import fit_grid_shape, load_model, save_model


def _data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    random = np.random.default_rng(345)
    states = np.zeros((121, 44))
    spot = random.normal(scale=.01, size=120)
    for index in range(1, len(states)):
        states[index] = .7 * states[index - 1] + random.normal(scale=.02, size=44) + spot[index - 1] * np.linspace(-1., 1., 44)
    return states[:-1], states[1:], spot


@pytest.mark.parametrize("orthogonal", [False, True])
def test_split_training_replay_restores_joint_spot_and_block_projection(orthogonal: bool) -> None:
    before, after, spot = _data()
    model = fit_grid_shape(before=before, after=after, spot=spot, weights=np.tile(np.r_[np.full(10, .1), 1.], 4),
        native_config=DlvConfig(business_dtes=(1., 2., 3., 5.), num_inner_strikes=10),
        config=DynamicsConfig(window_transitions=120, rank=6, spot_orthogonal=orthogonal), grid_rank=2,
        as_of_date="2024-04-01", training_start="2023-01-01", training_end="2024-04-01")
    np.testing.assert_allclose(model.encoder.T @ model.decoder, np.eye(6), atol=1e-12)
    changes = after - before
    scores = (before - model.native_mean) @ model.encoder
    replay = ((model.ar_intercept + scores @ model.ar_matrix.T - scores) @ model.decoder.T
        + model.residuals[:, :1] * model.native_spot_loading + model.residuals[:, 1:] @ model.decoder.T)
    np.testing.assert_allclose(replay, model.retained_changes(changes), atol=1e-12)
    np.testing.assert_array_equal(model.decoder[model.grid_indices, 2:], np.zeros((4, 4)))
    if orthogonal:
        expected = model.residuals[:, 0] @ (changes - changes.mean(axis=0)) / np.sum(model.residuals[:, 0]**2)
        np.testing.assert_allclose(model.native_spot_loading, expected, atol=1e-12)


def test_grid_shape_roundtrip_and_causal_orientation(tmp_path: Path) -> None:
    before, after, spot = _data()
    arguments = dict(before=before, after=after, spot=spot, weights=np.tile(np.r_[np.full(10, .1), 1.], 4),
        native_config=DlvConfig(business_dtes=(1., 2., 3., 5.), num_inner_strikes=10),
        config=DynamicsConfig(window_transitions=120, rank=6), grid_rank=2,
        training_start="2023-01-01", training_end="2024-04-01")
    model = fit_grid_shape(**arguments, as_of_date="2024-04-01")
    path = tmp_path / "model.npz"
    save_model(model, path)
    restored = load_model(path)
    np.testing.assert_array_equal(model.next_native(after[-1]), restored.next_native(after[-1]))
    following = fit_grid_shape(**arguments, as_of_date="2024-04-02", previous=model)
    assert following.initialisation_date == "2024-04-01"
