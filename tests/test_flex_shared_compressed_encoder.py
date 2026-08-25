"""Tests for frozen shared-strikes compressed decoder plumbing."""

from __future__ import annotations

import torch

from quantdata.dysanos.flex.config import (
    PcaDecoderConfig,
    SharedStrikeCappedUnboundedDlvFlexGeometryConfig,
)
from quantdata.dysanos.flex.market_encoder import (
    NativeRbfQuoteEncoder,
    NativeRbfQuoteEncoderConfig,
)
from quantdata.dysanos.flex.model import LatentStateSanosModel
from quantdata.dysanos.flex.run_shared_strikes_compressed_encoders import (
    _decoder_market_train_config,
    _install_frozen_decoder,
    _project_validation_states,
)
from quantdata.torch import BaseEnvironment


def test_frozen_direct_decoder_matches_and_projects_exactly() -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    geometry = SharedStrikeCappedUnboundedDlvFlexGeometryConfig(
        num_model_expiries=3,
        num_inner_strikes=6,
    )
    state_size = 4
    decoder = LatentStateSanosModel(
        env,
        num_days=2,
        config=PcaDecoderConfig(
            label="compressed-decoder-test",
            state_size=state_size,
            geometry=geometry,
        ),
    )
    with torch.no_grad():
        weight = decoder.raw_decoder.I0.weight
        weight.copy_(
            torch.linspace(
                -0.7,
                0.9,
                weight.numel(),
                dtype=env.fdtype,
            ).reshape_as(weight)
        )
        weight[:state_size].add_(torch.eye(state_size, dtype=env.fdtype))
        decoder.raw_decoder.I0.bias.copy_(
            torch.linspace(
                -0.2,
                0.3,
                geometry.native_state_size,
                dtype=env.fdtype,
            )
        )
    encoder = NativeRbfQuoteEncoder(
        env,
        config=NativeRbfQuoteEncoderConfig(
            label="compressed-encoder-test",
            state_size=state_size,
            geometry=geometry,
        ),
        native_mean=torch.zeros(geometry.native_state_size),
        native_scale=torch.ones(geometry.native_state_size),
        state_mean=torch.zeros(state_size),
        state_scale=torch.ones(state_size),
        pca_components=torch.zeros((state_size, geometry.native_state_size)),
    )
    _install_frozen_decoder(encoder, decoder)
    states = torch.tensor(
        [[-0.4, 0.1, 0.8, -0.2], [0.3, -0.5, 0.2, 0.7]],
        dtype=env.fdtype,
    )
    expected = decoder.raw_decoder(states)
    actual = encoder.native_raw_from_state(states)
    torch.testing.assert_close(actual, expected)
    projected = _project_validation_states(decoder, expected)
    torch.testing.assert_close(projected, states)
    assert encoder.native_decoder is not None
    assert not any(
        parameter.requires_grad for parameter in encoder.native_decoder.parameters()
    )


def test_incomplete_decoder_uses_explicit_low_lr_recovery(tmp_path) -> None:
    checkpoint = tmp_path / "incomplete.pt"
    torch.save({"schema_version": 2, "phase": "market"}, checkpoint)
    config = _decoder_market_train_config(
        checkpoint,
        num_epochs=1_000,
        learning_rate=3.0e-4,
        learning_rate_schedule="cosine",
        learning_rate_warmup_epochs=10,
        minimum_learning_rate_fraction=0.1,
        recover_incomplete=True,
        recovery_epochs=200,
        recovery_learning_rate=3.0e-5,
        device="cpu",
    )
    assert config.num_epochs == 200
    assert config.learning_rate == 3.0e-5
    assert config.learning_rate_schedule == "constant"
    assert config.learning_rate_warmup_epochs == 0
    assert config.minimum_learning_rate_fraction == 1.0
    assert config.warm_start is True
