from __future__ import annotations

from datetime import date

import pytest

from quantdata.hedging.protocols import BenchmarkConfig, EpisodeSpec, HedgeSchedule


def test_benchmark_defaults_encode_agreed_factor_and_step_counts() -> None:
    config = BenchmarkConfig(label="test")
    assert config.rolling_window_business_days == 252
    assert config.dlv_factor_rank == 16
    assert config.greek_factor_rank == 8
    assert config.brownian_substeps_per_day == 8
    assert config.nms_range == (-2.0, 2.0)


def test_frozen_weekly_refits_only_at_origin() -> None:
    episode = EpisodeSpec(
        schedule=HedgeSchedule.FROZEN_WEEKLY,
        origin_date=date(2025, 1, 6),
        training_dates=(date(2025, 1, 2), date(2025, 1, 3), date(2025, 1, 6)),
        decision_dates=(
            date(2025, 1, 6),
            date(2025, 1, 7),
            date(2025, 1, 8),
            date(2025, 1, 9),
            date(2025, 1, 10),
        ),
        terminal_date=date(2025, 1, 13),
        frozen_contract_ids=("a", "b"),
    )
    assert episode.refit_dates == (date(2025, 1, 6),)


def test_restarted_weekly_refits_each_decision_date() -> None:
    decisions = (date(2025, 1, 6), date(2025, 1, 7))
    episode = EpisodeSpec(
        schedule=HedgeSchedule.RESTARTED_WEEKLY,
        origin_date=decisions[0],
        training_dates=(date(2025, 1, 3), date(2025, 1, 6)),
        decision_dates=decisions,
        terminal_date=date(2025, 1, 8),
        frozen_contract_ids=("a",),
    )
    assert episode.refit_dates == decisions


def test_episode_rejects_lookahead_training_data() -> None:
    with pytest.raises(ValueError, match="cannot extend"):
        EpisodeSpec(
            schedule=HedgeSchedule.DAILY,
            origin_date=date(2025, 1, 6),
            training_dates=(date(2025, 1, 7),),
            decision_dates=(date(2025, 1, 6),),
            terminal_date=date(2025, 1, 7),
        )
