"""Floating-state decoding must round-trip and make numerical guards observable."""

from datetime import date

import numpy as np
import pytest

from quantdata.hedge2.dlv import DlvConfig, shared_inner_strikes
from quantdata.hedge2.state import DailyState


def _state() -> DailyState:
    config = DlvConfig(num_inner_strikes=20)
    vols = np.full(9, .2)
    inner = shared_inner_strikes(vols, config=config)
    return DailyState(day=date(2024, 4, 1), config=config,
                      native=np.c_[np.full((9, 20), np.log(.04)), np.log(vols)].reshape(-1),
                      reference_support=np.r_[.5 * inner[0], inner, 1.5 * inner[-1]])


def test_native_state_round_trip_and_floating_grid() -> None:
    state = _state()
    decoded = state.decode()
    np.testing.assert_allclose(decoded.pricer.support, state.reference_support)
    np.testing.assert_allclose(decoded.pricer.dlvsqs, .04)
    changed = state.native.reshape(9, 21).copy()
    changed[:, -1] += np.log(1.1)
    bumped = state.decode(changed.reshape(-1))
    np.testing.assert_allclose(np.log(bumped.pricer.support[1:-1]),
                               1.1 * np.log(decoded.pricer.support[1:-1]), atol=1e-15)
    assert decoded.lower_clips == decoded.upper_clips == 0


def test_native_guards_are_reported_and_extreme_geometry_is_rejected() -> None:
    state = _state()
    changed = state.native.copy()
    changed[0], changed[1] = -100., 100.
    decoded = state.decode(changed)
    assert decoded.lower_clips == decoded.upper_clips == 1
    changed.reshape(9, 21)[:, -1] = 100.
    with pytest.raises(ValueError, match="geometry"):
        state.decode(changed)


def test_states_not_built_from_zero_sanos_are_refused(tmp_path) -> None:
    import json
    from dataclasses import asdict

    configuration = asdict(DlvConfig())
    assert configuration["source_smoothness"] == 0.
    for stale in ({k: v for k, v in configuration.items() if k != "source_smoothness"},
                  {**configuration, "source_smoothness": .1}):
        (tmp_path / "config.json").write_text(json.dumps(stale))
        with pytest.raises(ValueError, match="0-SANOS"):
            DailyState.load(tmp_path, date(2024, 1, 2))


def test_dlv_build_refuses_an_archive_of_the_wrong_smoothness(tmp_path) -> None:
    from quantdata.hedge2.dlv import build_day
    from quantdata.hedge2.source import SourceArchive

    with pytest.raises(ValueError, match="smoothness-0 SANOS fit"):
        build_day(SourceArchive(root=tmp_path, smoothness=.1), date(2024, 1, 2), config=DlvConfig())
