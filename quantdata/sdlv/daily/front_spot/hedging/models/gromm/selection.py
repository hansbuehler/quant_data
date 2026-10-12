"""Generated 2022 data and reporting for GROMM architecture selection."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import numpy as np
from cdxcore.err import verify_inp
from cdxcore.version import version

from quantdata.sdlv.daily.front_spot.hedging.benchmark.digital_daily import (
    DailyDigitalTarget,
    daily_digital_targets,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import PublicHedgingPanel
from quantdata.sdlv.daily.front_spot.hedging.data.rolling_calibration import (
    RollingDlvCalibration,
    RollingDlvCalibrationConfig,
    fit_rolling_dlv_calibration,
)
from quantdata.sdlv.daily.front_spot.hedging.data.rolling_factors import (
    RollingDlvFactorConfig,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.config import (
    GrommConfig,
    GrommMeasure,
    GrommObjective,
    GrommTrainConfig,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.hedger import (
    GrommArchitectureSelection,
    select_gromm_architecture,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.policy import (
    gromm_architecture_candidates,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.simulation import (
    generate_gromm_daily_scenarios,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.training import (
    GrommTrainingData,
    daily_gromm_training_data,
)
from quantdata.torch import BaseEnvironment


@version("1.1", dependencies=[GrommConfig, GrommTrainConfig])
@dataclass(frozen=True, kw_only=True)
class GrommArchitectureStudyConfig:
    """Precommitted 2022 architecture-selection population."""

    origin_date: date = date(2022, 12, 30)
    calibration_window_sessions: int = 200
    paths_per_case: int = 128
    measures: tuple[GrommMeasure, ...] = (GrommMeasure.Q, GrommMeasure.P)
    objective: GrommObjective = GrommObjective.MEAN_VARIANCE
    risk_parameter: float = 10.0
    seed: int = 20_261_012
    position_cap: float = 1_000.0

    def __post_init__(self) -> None:
        verify_inp(
            self.origin_date.year == 2022,
            lambda: "architecture selection must use a 2022 origin",
        )
        verify_inp(
            self.calibration_window_sessions >= 32,
            lambda: "calibration_window_sessions must be at least 32",
        )
        verify_inp(
            self.paths_per_case >= 4 and self.paths_per_case % 2 == 0,
            lambda: "paths_per_case must be even and at least four",
        )
        verify_inp(
            self.measures == (GrommMeasure.Q, GrommMeasure.P),
            lambda: "architecture study must include Q then P",
        )
        verify_inp(
            math.isfinite(self.risk_parameter) and self.risk_parameter > 0.0,
            lambda: "risk_parameter must be finite-positive",
        )
        verify_inp(
            isinstance(self.seed, int) and not isinstance(self.seed, bool),
            lambda: "seed must be an integer",
        )
        verify_inp(
            math.isfinite(self.position_cap) and self.position_cap > 0.0,
            lambda: "position_cap must be finite-positive",
        )


@dataclass(frozen=True, kw_only=True)
class GrommArchitectureStudyData:
    """Combined generated cases and their shared rolling calibration."""

    training: GrommTrainingData
    calibration: RollingDlvCalibration
    case_labels: tuple[str, ...]
    paths_per_case: int


@dataclass(frozen=True, kw_only=True)
class GrommArchitectureStudyResult:
    """Architecture selection with an auditable JSON representation."""

    config: GrommArchitectureStudyConfig
    data: GrommArchitectureStudyData
    selection: GrommArchitectureSelection

    def to_dict(self) -> dict[str, object]:
        """Return the complete selection evidence as JSON-compatible values."""
        candidates = []
        for fit in self.selection.candidates:
            policy = fit.config
            candidates.append(
                {
                    "label": policy.label,
                    "hidden_size": policy.hidden_size,
                    "recurrent_layers": policy.recurrent_layers,
                    "deep_depth": policy.decision_network.num_depth,
                    "deep_width": policy.decision_network.num_width,
                    "parameter_count": fit.parameter_count,
                    "best_epoch": fit.result.best_epoch,
                    "completed_epochs": fit.result.completed_epochs,
                    "training_loss": fit.result.training_loss,
                    "validation_loss": fit.result.validation_loss,
                    "validation_standard_error": (
                        fit.result.validation_standard_error
                    ),
                    "cap_hit_fraction": fit.result.cap_hit_fraction,
                    "converged": fit.result.converged,
                }
            )
        return {
            "schema": "gromm-architecture-selection-v1",
            "origin_date": self.config.origin_date.isoformat(),
            "calibration_window_sessions": (
                self.config.calibration_window_sessions
            ),
            "factor_rank": self.data.calibration.factors.factor_rank,
            "factor_explained_variance_fraction": (
                self.data.calibration.factors.explained_variance_fraction
            ),
            "physical_drift_accepted": (
                self.data.calibration.rolling_physical is not None
                and self.data.calibration.rolling_physical.accepted
            ),
            "measures": [measure.value for measure in self.config.measures],
            "targets": [target.label for target in daily_digital_targets()],
            "case_count": len(self.data.case_labels),
            "paths_per_case": self.data.paths_per_case,
            "training_path_count": self.data.training.path_count,
            "instrument_count": self.data.training.instrument_count,
            "objective": self.config.objective.value,
            "risk_parameter": self.config.risk_parameter,
            "best_validation_loss": self.selection.best_validation_loss,
            "best_validation_standard_error": (
                self.selection.best_validation_standard_error
            ),
            "one_standard_error_threshold": (
                self.selection.best_validation_loss
                + self.selection.best_validation_standard_error
            ),
            "selected_label": self.selection.selected.label,
            "candidates": candidates,
        }

    def write_json(self, path: Path) -> None:
        """Write deterministic architecture-selection evidence."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )


def build_gromm_architecture_study_data(
    panel: PublicHedgingPanel,
    *,
    config: GrommArchitectureStudyConfig = GrommArchitectureStudyConfig(),
) -> GrommArchitectureStudyData:
    """Generate all daily digital cases under both Q and accepted P."""
    calibration_config = RollingDlvCalibrationConfig(
        factor=RollingDlvFactorConfig(
            window_sessions=config.calibration_window_sessions,
            factor_rank=16,
        ),
        greek_factor_count=8,
    )
    calibration = fit_rolling_dlv_calibration(
        panel,
        as_of_date=config.origin_date,
        config=calibration_config,
    )
    if calibration.rolling_physical is None or not calibration.rolling_physical.accepted:
        raise RuntimeError("2022 architecture study requires an accepted P drift")
    cases: list[GrommTrainingData] = []
    labels: list[str] = []
    targets = daily_digital_targets()
    for measure_index, measure in enumerate(config.measures):
        for target_index, target in enumerate(targets):
            case_seed = config.seed + measure_index * len(targets) + target_index
            model_config = GrommConfig(
                label="gromm-architecture-2022",
                calibration=calibration_config,
                measure=measure,
                objective=config.objective,
                risk_parameter=config.risk_parameter,
                path_count=config.paths_per_case,
                batch_size=config.paths_per_case,
                seed=case_seed,
                position_cap=config.position_cap,
            )
            scenarios = generate_gromm_daily_scenarios(
                panel,
                origin_date=config.origin_date,
                target=target,
                config=model_config,
                calibration=calibration,
            )
            cases.append(daily_gromm_training_data(scenarios))
            labels.append(f"{measure.value}:{target.label}")
    return GrommArchitectureStudyData(
        training=combine_gromm_training_data(tuple(cases)),
        calibration=calibration,
        case_labels=tuple(labels),
        paths_per_case=config.paths_per_case,
    )


def combine_gromm_training_data(
    cases: tuple[GrommTrainingData, ...],
) -> GrommTrainingData:
    """Combine independent antithetic case blocks with one instrument geometry."""
    if not cases:
        raise ValueError("at least one GROMM architecture case is required")
    reference = cases[0]
    for case in cases:
        if case.path_count % 2:
            raise ValueError("architecture cases must preserve antithetic pairs")
        if (
            case.feature_count != reference.feature_count
            or case.instrument_count != reference.instrument_count
            or not np.array_equal(case.active_mask, reference.active_mask)
            or not np.array_equal(
                case.linear_transaction_costs,
                reference.linear_transaction_costs,
            )
        ):
            raise ValueError("architecture cases do not share policy geometry")
    return GrommTrainingData(
        features=np.concatenate([case.features for case in cases], axis=0),
        instrument_returns=np.concatenate(
            [case.instrument_returns for case in cases], axis=0
        ),
        claim_returns=np.concatenate(
            [case.claim_returns for case in cases], axis=0
        ),
        active_mask=reference.active_mask.copy(),
        linear_transaction_costs=reference.linear_transaction_costs.copy(),
    )


def run_gromm_architecture_study(
    panel: PublicHedgingPanel,
    *,
    checkpoint_directory: Path,
    config: GrommArchitectureStudyConfig = GrommArchitectureStudyConfig(),
    train_config: GrommTrainConfig = GrommTrainConfig(),
    environment: BaseEnvironment | None = None,
) -> GrommArchitectureStudyResult:
    """Generate the 2022 study and apply the one-standard-error rule."""
    data = build_gromm_architecture_study_data(panel, config=config)
    candidates = gromm_architecture_candidates(
        label_prefix="gromm-architecture-2022",
        feature_count=data.training.feature_count,
        instrument_count=data.training.instrument_count,
        position_cap=config.position_cap,
    )
    selection = select_gromm_architecture(
        data.training,
        candidates=candidates,
        objective=config.objective,
        risk_parameter=config.risk_parameter,
        checkpoint_directory=checkpoint_directory,
        train_config=replace(train_config, seed=config.seed),
        environment=environment,
    )
    return GrommArchitectureStudyResult(
        config=config,
        data=data,
        selection=selection,
    )
