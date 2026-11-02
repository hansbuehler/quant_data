from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from quantdata.dysanos.flex.config import FlexObjectiveConfig
from quantdata.dysanos.v11.run_rolling_direct_ar import _date_evaluation, _fit_config


def test_native_and_decoder_keep_the_established_separate_training_rates() -> None:
    native = _fit_config(epochs=1000, device="cpu", stage="native")
    decoder = _fit_config(epochs=1000, device="cpu", stage="decoder")
    assert native.learning_rate == pytest.approx(0.02)
    assert native.decoder_learning_rate is None
    assert decoder.learning_rate == pytest.approx(0.0001)
    assert decoder.decoder_learning_rate == pytest.approx(0.0001)
    for config in (native, decoder):
        assert config.day_batch_size == 128
        assert config.minimum_learning_rate_fraction == 0.0


def test_single_date_report_selects_its_own_prices_and_quotes() -> None:
    first = SimpleNamespace(
        bids=np.array([0.10]), asks=np.array([0.12]), fit_prices=np.array([0.11]),
        quote_business_dtes=np.array([1.0]),
    )
    last = SimpleNamespace(
        bids=np.array([0.30, 0.40]), asks=np.array([0.32, 0.42]),
        fit_prices=np.array([0.31, 0.41]), quote_business_dtes=np.array([1.0, 2.0]),
    )
    evaluation = SimpleNamespace(fits=(np.array([0.11]), np.array([0.31, 0.44])))
    result = _date_evaluation(
        evaluation, (first, last), index=-1, objective=FlexObjectiveConfig(fit_weight=0.05),
    )
    assert result.statistics.all.num_options == 2
    assert result.statistics.all.percent_zero == 50.0
    assert result.statistics.all.positive_mean == pytest.approx(2.0)
    assert result.objective == pytest.approx(1.075)


def test_single_date_report_rejects_unaligned_date_sequences() -> None:
    with pytest.raises(ValueError, match="same length"):
        _date_evaluation(
            SimpleNamespace(fits=()), (SimpleNamespace(),), index=-1,
            objective=FlexObjectiveConfig(),
        )
