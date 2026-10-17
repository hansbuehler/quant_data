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
from quantdata.sdlv.daily.front_spot.hedging.models.checkpointing import file_sha256
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
    GrommPolicyArchitecture,
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


@version(
    "1.2",
    dependencies=[GrommConfig, GrommTrainConfig, GrommPolicyArchitecture],
)
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
    dataset_manifest_sha256: str


@dataclass(frozen=True, kw_only=True)
class GrommArchitectureSelectionArtifact:
    """Authenticated 2022 architecture frozen for rolling production."""

    path: Path
    sha256: str
    dataset_manifest_sha256: str
    selected_label: str
    architecture: GrommPolicyArchitecture
    training_feature_count: int
    training_instrument_count: int


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
                    "deep_activation": policy.decision_network.activation,
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
            "schema": "gromm-architecture-selection-v2",
            "code_version": str(type(self.config).version),
            "dataset_manifest_sha256": self.data.dataset_manifest_sha256,
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
            "training_feature_count": self.data.training.feature_count,
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
        dataset_manifest_sha256=panel.dataset_manifest_sha256,
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
            or case.mean_reduction is not reference.mean_reduction
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
        scenario_weights=np.concatenate(
            [case.normalized_scenario_weights for case in cases], axis=0
        )
        / len(cases),
        mean_reduction=reference.mean_reduction,
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


def load_gromm_architecture_selection(
    path: Path,
    *,
    expected_dataset_manifest_sha256: str,
) -> GrommArchitectureSelectionArtifact:
    """Load and authenticate the frozen 2022 GROMM architecture decision."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "gromm-architecture-selection-v2":
        raise ValueError("GROMM architecture selection has an incompatible schema")
    if payload.get("code_version") != str(GrommArchitectureStudyConfig.version):
        raise ValueError("GROMM architecture selection has an incompatible code version")
    dataset_manifest_sha256 = _required_sha256(
        payload,
        "dataset_manifest_sha256",
    )
    if dataset_manifest_sha256 != expected_dataset_manifest_sha256:
        raise ValueError("GROMM architecture selection belongs to another dataset")
    origin_date = date.fromisoformat(_required_string(payload, "origin_date"))
    if origin_date.year != 2022:
        raise ValueError("GROMM architecture selection must use a 2022 origin")
    if payload.get("measures") != ["Q", "P"]:
        raise ValueError("GROMM architecture selection must contain Q then P")
    expected_targets = [target.label for target in daily_digital_targets()]
    if payload.get("targets") != expected_targets:
        raise ValueError("GROMM architecture selection has the wrong target book")
    if payload.get("case_count") != 2 * len(expected_targets):
        raise ValueError("GROMM architecture selection has the wrong case count")
    if payload.get("factor_rank") != 16:
        raise ValueError("GROMM architecture selection must use 16 factors")
    if payload.get("physical_drift_accepted") is not True:
        raise ValueError("GROMM architecture selection requires an accepted P drift")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 4:
        raise ValueError("GROMM architecture selection must contain four candidates")
    if any(not isinstance(candidate, dict) for candidate in candidates):
        raise ValueError("GROMM architecture candidates must be JSON objects")
    labels = tuple(_required_string(candidate, "label") for candidate in candidates)
    if len(set(labels)) != len(labels):
        raise ValueError("GROMM architecture candidate labels must be unique")
    losses = tuple(
        _required_finite_float(candidate, "validation_loss")
        for candidate in candidates
    )
    standard_errors = tuple(
        _required_nonnegative_float(candidate, "validation_standard_error")
        for candidate in candidates
    )
    best_index = min(range(len(candidates)), key=losses.__getitem__)
    best_loss = _required_finite_float(payload, "best_validation_loss")
    best_standard_error = _required_nonnegative_float(
        payload,
        "best_validation_standard_error",
    )
    if (
        not math.isclose(best_loss, losses[best_index], rel_tol=0.0, abs_tol=1.0e-14)
        or not math.isclose(
            best_standard_error,
            standard_errors[best_index],
            rel_tol=0.0,
            abs_tol=1.0e-14,
        )
    ):
        raise ValueError("GROMM architecture best-candidate statistics do not match")
    threshold = _required_finite_float(payload, "one_standard_error_threshold")
    if not math.isclose(
        threshold,
        best_loss + best_standard_error,
        rel_tol=0.0,
        abs_tol=1.0e-14,
    ):
        raise ValueError("GROMM architecture one-SE threshold does not match")
    selected_label = _required_string(payload, "selected_label")
    selected = tuple(
        candidate
        for candidate in candidates
        if isinstance(candidate, dict) and candidate.get("label") == selected_label
    )
    if len(selected) != 1:
        raise ValueError("GROMM selected architecture is not unique")
    candidate = selected[0]
    if candidate.get("converged") is not True:
        raise ValueError("GROMM selected architecture did not converge")
    cap_hit_fraction = _required_finite_float(candidate, "cap_hit_fraction")
    if cap_hit_fraction != 0.0:
        raise ValueError("GROMM selected architecture has position-cap hits")
    eligible = tuple(
        current
        for current, loss in zip(candidates, losses, strict=True)
        if loss <= threshold
    )
    expected_selected = min(
        eligible,
        key=lambda current: (
            _required_positive_int(current, "parameter_count"),
            _required_finite_float(current, "validation_loss"),
        ),
    )
    if candidate is not expected_selected:
        raise ValueError("GROMM selected architecture violates the one-SE rule")
    architecture = GrommPolicyArchitecture(
        hidden_size=_required_positive_int(candidate, "hidden_size"),
        recurrent_layers=_required_positive_int(candidate, "recurrent_layers"),
        deep_activation=_required_string(candidate, "deep_activation"),
        deep_depth=_required_positive_int(candidate, "deep_depth"),
        deep_width=_required_positive_int(candidate, "deep_width"),
    )
    return GrommArchitectureSelectionArtifact(
        path=path.resolve(),
        sha256=file_sha256(path),
        dataset_manifest_sha256=dataset_manifest_sha256,
        selected_label=selected_label,
        architecture=architecture,
        training_feature_count=_required_positive_int(
            payload,
            "training_feature_count",
        ),
        training_instrument_count=_required_positive_int(
            payload,
            "instrument_count",
        ),
    )


def _required_string(payload: dict[str, object], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"GROMM architecture field {name!r} must be non-empty")
    return value


def _required_sha256(payload: dict[str, object], name: str) -> str:
    value = _required_string(payload, name).lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"GROMM architecture field {name!r} must be a SHA-256")
    return value


def _required_positive_int(payload: dict[str, object], name: str) -> int:
    value = payload.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"GROMM architecture field {name!r} must be positive")
    return value


def _required_finite_float(payload: dict[str, object], name: str) -> float:
    value = payload.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"GROMM architecture field {name!r} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"GROMM architecture field {name!r} must be finite")
    return numeric


def _required_nonnegative_float(payload: dict[str, object], name: str) -> float:
    value = _required_finite_float(payload, name)
    if value < 0.0:
        raise ValueError(f"GROMM architecture field {name!r} must be non-negative")
    return value
