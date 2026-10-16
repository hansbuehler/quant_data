"""Pre-outcome weekly Historical-SANOS ridge calibration."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from quantdata.hedging.protocols import HedgeSchedule
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly import (
    WeeklyClaimDefinition,
    WeeklyFitInstruction,
    weekly_inception_quotes,
)
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly_runner import (
    WeeklyDecisionState,
)
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly_scenarios import (
    WeeklyRiskObjective,
    WeeklyScenarioBatch,
    weekly_certainty_equivalent,
    weekly_scenario_accounting,
)
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly_sizing import (
    WeeklyTargetNotionalCalibration,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    PublicHedgingPanel,
)
from quantdata.sdlv.daily.front_spot.hedging.models.historical_sanos.config import (
    HistoricalSanosConfig,
    HistoricalSanosObjective,
)
from quantdata.sdlv.daily.front_spot.hedging.models.historical_sanos.weekly_hedger import (
    HistoricalWeeklyOpenLoopFit,
    WeeklyOpenLoopPricingContext,
    fit_historical_weekly_hedge,
)
from quantdata.sdlv.daily.front_spot.hedging.models.historical_sanos.weekly_scenarios import (
    HistoricalWeeklyScenarioSet,
    HistoricalWeeklyTransitionHistory,
    attach_historical_weekly_claim,
    build_historical_weekly_scenarios,
)
from quantdata.sdlv.daily.front_spot.hedging.models.sanos_lv.weekly_pricing import (
    build_common_weekly_claims,
)

FloatArray = NDArray[np.float64]
ProgressFunction = Callable[[str], None]

WEEKLY_RIDGE_SELECTION_SCHEMA = "hedging-historical-sanos-weekly-ridge-v4"
WEEKLY_RIDGE_BUNDLE_SCHEMA = "hedging-historical-sanos-weekly-ridge-bundle-v2"
WEEKLY_RIDGE_SCHEDULE_FAMILY = (
    HedgeSchedule.FROZEN_WEEKLY.value,
    HedgeSchedule.RESTARTED_WEEKLY.value,
)
DEFAULT_WEEKLY_RIDGE_CANDIDATES = (
    1.0e-4,
    1.0e-3,
    1.0e-2,
    1.0e-1,
    1.0,
    10.0,
    100.0,
    1_000.0,
)
DEFAULT_CALIBRATION_ORIGIN = date(2023, 1, 17)
DEFAULT_WEEKLY_POSITION_CAP = 1_000.0


@dataclass(frozen=True, kw_only=True)
class WeeklyRidgeCandidateResult:
    """Held-out result and certification gates for one ridge candidate."""

    ridge: float | None
    block_regrets: tuple[float, ...]
    pricing_block_regrets: tuple[float, ...]
    zero_block_regrets: tuple[float, ...]
    execution_block_regrets: tuple[float, ...]
    mean_regret: float
    standard_error: float
    feasible: bool
    selection_converged_components: int
    final_converged_components: int
    converged_components: int
    component_count: int
    selection_cap_hits: int
    internal_cap_hits: int
    executable_cap_hits: int
    cap_hits: int
    unregularized_recession_directions: int
    recession_directions: int
    maximum_objective_error: float
    maximum_constraint_violation: float
    maximum_latent_coordinate: float

    def __post_init__(self) -> None:
        if self.ridge is not None and (
            not math.isfinite(self.ridge) or self.ridge <= 0.0
        ):
            raise ValueError("weekly ridge candidate must be finite-positive or null")
        regret_vectors = (
            self.block_regrets,
            self.pricing_block_regrets,
            self.zero_block_regrets,
            self.execution_block_regrets,
        )
        if len(self.block_regrets) < 2 or any(
            len(values) != len(self.block_regrets)
            or not all(math.isfinite(value) for value in values)
            for values in regret_vectors
        ):
            raise ValueError("weekly ridge block regrets must be finite and aligned")
        if not math.isfinite(self.mean_regret) or not math.isfinite(
            self.standard_error
        ):
            raise ValueError("weekly ridge summary statistics must be finite")
        if self.standard_error < 0.0:
            raise ValueError("weekly ridge standard error cannot be negative")
        expected_regrets = tuple(
            (pricing + zero + execution) / 3.0
            for pricing, zero, execution in zip(
                self.pricing_block_regrets,
                self.zero_block_regrets,
                self.execution_block_regrets,
                strict=True,
            )
        )
        expected_mean = float(np.mean(expected_regrets))
        expected_standard_error = float(
            np.std(expected_regrets, ddof=1) / math.sqrt(len(expected_regrets))
        )
        if not all(
            math.isclose(actual, expected, rel_tol=1.0e-12, abs_tol=1.0e-14)
            for actual, expected in zip(
                self.block_regrets,
                expected_regrets,
                strict=True,
            )
        ):
            raise ValueError("weekly ridge aggregate regrets are inconsistent")
        if not math.isclose(
            self.mean_regret,
            expected_mean,
            rel_tol=1.0e-12,
            abs_tol=1.0e-14,
        ) or not math.isclose(
            self.standard_error,
            expected_standard_error,
            rel_tol=1.0e-12,
            abs_tol=1.0e-14,
        ):
            raise ValueError("weekly ridge regret statistics are inconsistent")
        if (
            not 0 <= self.selection_converged_components * 2 <= self.component_count
            or not 0 <= self.final_converged_components * 2 <= self.component_count
            or self.converged_components
            != self.selection_converged_components + self.final_converged_components
            or not 0 <= self.converged_components <= self.component_count
        ):
            raise ValueError("weekly ridge component counts are invalid")
        if (
            self.component_count <= 0
            or self.component_count % 2
            or self.selection_cap_hits < 0
            or self.internal_cap_hits < 0
            or self.executable_cap_hits < 0
            or self.cap_hits < 0
            or self.unregularized_recession_directions < 0
            or self.recession_directions < 0
        ):
            raise ValueError("weekly ridge diagnostics are invalid")
        if (
            self.internal_cap_hits < self.selection_cap_hits
            or self.cap_hits != self.internal_cap_hits + self.executable_cap_hits
        ):
            raise ValueError("weekly ridge cap-hit counts are inconsistent")
        if self.feasible and (
            self.converged_components != self.component_count
            or self.cap_hits
            or self.recession_directions
        ):
            raise ValueError("weekly ridge feasible candidate has failed gates")
        non_negative = (
            self.maximum_objective_error,
            self.maximum_constraint_violation,
            self.maximum_latent_coordinate,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in non_negative):
            raise ValueError("weekly ridge certificate values are invalid")


@dataclass(frozen=True, kw_only=True)
class WeeklyRidgeSelection:
    """Authenticated pre-outcome ridge selection for weekly historical hedges."""

    dataset_manifest_sha256: str
    target_notional_artifact_sha256: str
    origin_date: date
    training_origin_dates: tuple[date, ...]
    scenario_origin_dates: tuple[date, ...]
    fit_origin_dates: tuple[date, ...]
    validation_origin_dates: tuple[date, ...]
    validation_blocks: tuple[tuple[date, ...], ...]
    schedule_family: tuple[str, ...]
    target_labels: tuple[str, ...]
    target_quantities: tuple[float, ...]
    objective: HistoricalSanosObjective
    risk_parameter: float
    validation_fraction: float
    position_cap: float
    maximum_latent_coordinate: float
    candidates: tuple[WeeklyRidgeCandidateResult, ...]
    selected_ridge: float | None
    selected_by_one_standard_error: bool
    source_commit: str
    code_version: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.dataset_manifest_sha256, "dataset manifest"),
            (self.target_notional_artifact_sha256, "target-notional artifact"),
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"weekly ridge {name} must be a SHA-256")
        if len(self.training_origin_dates) != 252:
            raise ValueError("weekly ridge calibration requires 252 training origins")
        date_vectors = (
            self.training_origin_dates,
            self.scenario_origin_dates,
            self.fit_origin_dates,
            self.validation_origin_dates,
        )
        if any(
            not values
            or values != tuple(sorted(values))
            or len(set(values)) != len(values)
            for values in date_vectors
        ):
            raise ValueError("weekly ridge date vectors must be nonempty and chronological")
        if self.training_origin_dates[-1] >= self.origin_date:
            raise ValueError("weekly ridge calibration cannot use its outcome origin")
        if (
            tuple(
                value
                for value in self.training_origin_dates
                if value in set(self.scenario_origin_dates)
            )
            != self.scenario_origin_dates
            or self.scenario_origin_dates
            != self.fit_origin_dates + self.validation_origin_dates
        ):
            raise ValueError("weekly ridge scenario dates must partition into fit and validation")
        if (
            len(self.validation_blocks) != 5
            or any(not block for block in self.validation_blocks)
            or tuple(value for block in self.validation_blocks for value in block)
            != self.validation_origin_dates
        ):
            raise ValueError("weekly ridge validation blocks must partition validation dates")
        expected_validation_count = max(
            1,
            int(round(len(self.scenario_origin_dates) * self.validation_fraction)),
        )
        if (
            not 0.0 < self.validation_fraction < 1.0
            or len(self.validation_origin_dates) != expected_validation_count
            or len(self.fit_origin_dates) < 2
        ):
            raise ValueError("weekly ridge validation fraction is inconsistent")
        if not math.isfinite(self.position_cap) or self.position_cap <= 0.0:
            raise ValueError("weekly ridge position cap must be finite-positive")
        if self.schedule_family != WEEKLY_RIDGE_SCHEDULE_FAMILY:
            raise ValueError("weekly ridge schedule family is incompatible")
        if len(self.target_labels) != len(self.target_quantities) or not all(
            math.isfinite(value) and value > 0.0 for value in self.target_quantities
        ):
            raise ValueError("weekly ridge target quantities are invalid")
        if not self.candidates or len({value.ridge for value in self.candidates}) != len(
            self.candidates
        ):
            raise ValueError("weekly ridge candidates must be unique")
        selected = tuple(
            candidate
            for candidate in self.candidates
            if candidate.ridge == self.selected_ridge
        )
        if len(selected) != 1 or not selected[0].feasible:
            raise ValueError("weekly selected ridge must identify one feasible candidate")
        if not self.selected_by_one_standard_error:
            raise ValueError("weekly ridge selection must use the one-SE rule")
        if select_weekly_ridge(self.candidates).ridge != self.selected_ridge:
            raise ValueError("weekly ridge selected policy violates the one-SE rule")
        if not self.code_version.strip():
            raise ValueError("weekly ridge selection requires a code version")
        if len(self.source_commit) != 40 or any(
            character not in "0123456789abcdef" for character in self.source_commit
        ):
            raise ValueError("weekly ridge source commit must be a Git SHA-1")

    @property
    def sha256(self) -> str:
        """Return the stable digest of the selection payload."""
        return hashlib.sha256(_canonical_json(self.to_dict()).encode()).hexdigest()

    def to_dict(self) -> dict[str, object]:
        """Return a portable representation of the complete selection."""
        return {
            "schema": WEEKLY_RIDGE_SELECTION_SCHEMA,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "target_notional_artifact_sha256": self.target_notional_artifact_sha256,
            "origin_date": self.origin_date.isoformat(),
            "training_origin_dates": [value.isoformat() for value in self.training_origin_dates],
            "scenario_origin_dates": [value.isoformat() for value in self.scenario_origin_dates],
            "fit_origin_dates": [value.isoformat() for value in self.fit_origin_dates],
            "validation_origin_dates": [
                value.isoformat() for value in self.validation_origin_dates
            ],
            "validation_blocks": [
                [value.isoformat() for value in block]
                for block in self.validation_blocks
            ],
            "schedule_family": list(self.schedule_family),
            "target_labels": list(self.target_labels),
            "target_quantities": list(self.target_quantities),
            "objective": self.objective.value,
            "risk_parameter": self.risk_parameter,
            "validation_fraction": self.validation_fraction,
            "position_cap": self.position_cap,
            "maximum_latent_coordinate": self.maximum_latent_coordinate,
            "candidates": [asdict(candidate) for candidate in self.candidates],
            "selected_ridge": self.selected_ridge,
            "selected_policy": (
                "null-baseline" if self.selected_ridge is None else "ridge"
            ),
            "selected_by_one_standard_error": self.selected_by_one_standard_error,
            "source_commit": self.source_commit,
            "code_version": self.code_version,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> WeeklyRidgeSelection:
        """Reconstruct and validate a portable selection artifact."""
        if payload.get("schema") != WEEKLY_RIDGE_SELECTION_SCHEMA:
            raise ValueError("weekly ridge artifact has an incompatible schema")
        try:
            candidates_value = payload["candidates"]
            if not isinstance(candidates_value, list):
                raise TypeError("candidates must be a list")
            return cls(
                dataset_manifest_sha256=str(payload["dataset_manifest_sha256"]),
                target_notional_artifact_sha256=str(
                    payload["target_notional_artifact_sha256"]
                ),
                origin_date=date.fromisoformat(str(payload["origin_date"])),
                training_origin_dates=tuple(
                    date.fromisoformat(str(value))
                    for value in payload["training_origin_dates"]
                ),
                scenario_origin_dates=tuple(
                    date.fromisoformat(str(value))
                    for value in payload["scenario_origin_dates"]
                ),
                fit_origin_dates=tuple(
                    date.fromisoformat(str(value))
                    for value in payload["fit_origin_dates"]
                ),
                validation_origin_dates=tuple(
                    date.fromisoformat(str(value))
                    for value in payload["validation_origin_dates"]
                ),
                validation_blocks=tuple(
                    tuple(date.fromisoformat(str(value)) for value in block)
                    for block in payload["validation_blocks"]
                ),
                schedule_family=tuple(
                    str(value) for value in payload["schedule_family"]
                ),
                target_labels=tuple(str(value) for value in payload["target_labels"]),
                target_quantities=tuple(
                    float(value) for value in payload["target_quantities"]
                ),
                objective=HistoricalSanosObjective(str(payload["objective"])),
                risk_parameter=float(payload["risk_parameter"]),
                validation_fraction=float(payload["validation_fraction"]),
                position_cap=float(payload["position_cap"]),
                maximum_latent_coordinate=float(payload["maximum_latent_coordinate"]),
                candidates=tuple(
                    WeeklyRidgeCandidateResult(
                        ridge=(
                            None
                            if value["ridge"] is None
                            else float(value["ridge"])
                        ),
                        block_regrets=tuple(
                            float(item) for item in value["block_regrets"]
                        ),
                        pricing_block_regrets=tuple(
                            float(item)
                            for item in value["pricing_block_regrets"]
                        ),
                        zero_block_regrets=tuple(
                            float(item) for item in value["zero_block_regrets"]
                        ),
                        execution_block_regrets=tuple(
                            float(item)
                            for item in value["execution_block_regrets"]
                        ),
                        mean_regret=float(value["mean_regret"]),
                        standard_error=float(value["standard_error"]),
                        feasible=bool(value["feasible"]),
                        selection_converged_components=int(
                            value["selection_converged_components"]
                        ),
                        final_converged_components=int(
                            value["final_converged_components"]
                        ),
                        converged_components=int(value["converged_components"]),
                        component_count=int(value["component_count"]),
                        selection_cap_hits=int(value["selection_cap_hits"]),
                        internal_cap_hits=int(value["internal_cap_hits"]),
                        executable_cap_hits=int(value["executable_cap_hits"]),
                        cap_hits=int(value["cap_hits"]),
                        unregularized_recession_directions=int(
                            value["unregularized_recession_directions"]
                        ),
                        recession_directions=int(value["recession_directions"]),
                        maximum_objective_error=float(
                            value["maximum_objective_error"]
                        ),
                        maximum_constraint_violation=float(
                            value["maximum_constraint_violation"]
                        ),
                        maximum_latent_coordinate=float(
                            value["maximum_latent_coordinate"]
                        ),
                    )
                    for value in candidates_value
                    if isinstance(value, dict)
                ),
                selected_ridge=(
                    None
                    if payload["selected_ridge"] is None
                    else float(payload["selected_ridge"])
                ),
                selected_by_one_standard_error=bool(
                    payload["selected_by_one_standard_error"]
                ),
                source_commit=str(payload["source_commit"]),
                code_version=str(payload["code_version"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("weekly ridge artifact is incomplete") from error


@dataclass(frozen=True, kw_only=True)
class WeeklyRidgeBundle:
    """Authenticated objective/risk-specific weekly ridge selections."""

    dataset_manifest_sha256: str
    target_notional_artifact_sha256: str
    quant_review_sha256: str
    selections: tuple[WeeklyRidgeSelection, ...]

    def __post_init__(self) -> None:
        for value, name in (
            (self.dataset_manifest_sha256, "dataset manifest"),
            (self.target_notional_artifact_sha256, "target-notional artifact"),
            (self.quant_review_sha256, "quant review"),
        ):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"weekly ridge bundle {name} must be a SHA-256")
        if not self.selections:
            raise ValueError("weekly ridge bundle requires selections")
        identities = tuple(
            (selection.objective, selection.risk_parameter)
            for selection in self.selections
        )
        if len(set(identities)) != len(identities):
            raise ValueError("weekly ridge bundle scopes must be unique")
        if any(
            selection.dataset_manifest_sha256 != self.dataset_manifest_sha256
            or selection.target_notional_artifact_sha256
            != self.target_notional_artifact_sha256
            for selection in self.selections
        ):
            raise ValueError("weekly ridge bundle identities do not align")
        reference_design = _bundle_design_identity(self.selections[0])
        if any(
            _bundle_design_identity(selection) != reference_design
            for selection in self.selections[1:]
        ):
            raise ValueError("weekly ridge bundle calibration designs do not align")

    @property
    def sha256(self) -> str:
        """Return the stable digest of the bundle payload."""
        return hashlib.sha256(_canonical_json(self.to_dict()).encode()).hexdigest()

    def to_dict(self) -> dict[str, object]:
        """Return a portable representation of every calibrated scope."""
        return {
            "schema": WEEKLY_RIDGE_BUNDLE_SCHEMA,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "target_notional_artifact_sha256": (
                self.target_notional_artifact_sha256
            ),
            "quant_review_sha256": self.quant_review_sha256,
            "selections": [
                selection.to_dict()
                for selection in sorted(
                    self.selections,
                    key=lambda item: (item.objective.value, item.risk_parameter),
                )
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> WeeklyRidgeBundle:
        """Reconstruct and validate one complete ridge bundle."""
        if payload.get("schema") != WEEKLY_RIDGE_BUNDLE_SCHEMA:
            raise ValueError("weekly ridge bundle has an incompatible schema")
        selections_value = payload.get("selections")
        if not isinstance(selections_value, list) or any(
            not isinstance(value, dict) for value in selections_value
        ):
            raise ValueError("weekly ridge bundle selections must be objects")
        return cls(
            dataset_manifest_sha256=str(payload.get("dataset_manifest_sha256")),
            target_notional_artifact_sha256=str(
                payload.get("target_notional_artifact_sha256")
            ),
            quant_review_sha256=str(payload.get("quant_review_sha256")),
            selections=tuple(
                WeeklyRidgeSelection.from_dict(value)
                for value in selections_value
            ),
        )

    def selection_for(
        self,
        *,
        objective: HistoricalSanosObjective,
        risk_parameter: float,
    ) -> WeeklyRidgeSelection:
        """Return the unique calibrated selection for one objective/risk pair."""
        matches = tuple(
            selection
            for selection in self.selections
            if selection.objective is objective
            and math.isclose(
                selection.risk_parameter,
                risk_parameter,
                rel_tol=0.0,
                abs_tol=1.0e-14,
            )
        )
        if len(matches) != 1:
            raise ValueError(
                "weekly ridge bundle lacks the requested objective/risk scope"
            )
        return matches[0]


def run_weekly_ridge_calibration(
    panel: PublicHedgingPanel,
    *,
    target_notionals: WeeklyTargetNotionalCalibration,
    target_notional_artifact_sha256: str,
    origin_date: date = DEFAULT_CALIBRATION_ORIGIN,
    ridge_candidates: tuple[float, ...] = DEFAULT_WEEKLY_RIDGE_CANDIDATES,
    objective: HistoricalSanosObjective = HistoricalSanosObjective.ENTROPY,
    risk_parameter: float = 0.1,
    position_cap: float = DEFAULT_WEEKLY_POSITION_CAP,
    maximum_latent_coordinate: float = 1.0e5,
    source_commit: str,
    progress: ProgressFunction | None = None,
) -> WeeklyRidgeSelection:
    """Select the weekly ridge using only information available at one origin."""
    if origin_date not in panel.dates:
        raise ValueError("weekly ridge origin is absent from the public panel")
    if not ridge_candidates or any(
        not math.isfinite(value) or value <= 0.0 for value in ridge_candidates
    ):
        raise ValueError("weekly ridge candidates must be finite-positive")
    if len(set(ridge_candidates)) != len(ridge_candidates):
        raise ValueError("weekly ridge candidates must be unique")
    if not math.isfinite(maximum_latent_coordinate) or maximum_latent_coordinate <= 0.0:
        raise ValueError("maximum latent coordinate must be finite-positive")
    if not math.isfinite(position_cap) or position_cap <= 0.0:
        raise ValueError("position cap must be finite-positive")

    base_config = HistoricalSanosConfig(
        label="historical-sanos-weekly-ridge-calibration",
        objective=objective,
        risk_parameter=risk_parameter,
        position_cap=position_cap,
    )
    prepared = _prepare_origin(
        panel,
        target_notionals=target_notionals,
        origin_date=origin_date,
        validation_fraction=base_config.validation_fraction,
    )
    candidate_results: list[WeeklyRidgeCandidateResult] = []
    started = time.monotonic()
    for candidate_index, ridge in enumerate(ridge_candidates, start=1):
        _emit(
            progress,
            f"weekly ridge candidate {candidate_index}/{len(ridge_candidates)} "
            f"ridge={ridge:g} started elapsed_s={time.monotonic() - started:.1f}",
        )
        candidate_results.append(
            _evaluate_candidate(
                prepared,
                config=replace(
                    base_config,
                    label=f"historical-sanos-weekly-ridge-{ridge:g}",
                    quadratic_ridge=ridge,
                ),
                maximum_latent_coordinate=maximum_latent_coordinate,
                progress=progress,
                candidate_index=candidate_index,
                candidate_count=len(ridge_candidates),
                started=started,
            )
        )
    candidate_results.append(
        _null_candidate_result(
            block_count=len(prepared.validation_blocks),
            component_count=3 * len(prepared.targets) + 1,
        )
    )
    _emit(
        progress,
        "weekly ridge null-policy candidate added "
        f"elapsed_s={time.monotonic() - started:.1f}",
    )
    selected = select_weekly_ridge(tuple(candidate_results))
    return WeeklyRidgeSelection(
        dataset_manifest_sha256=panel.dataset_manifest_sha256,
        target_notional_artifact_sha256=target_notional_artifact_sha256,
        origin_date=origin_date,
        training_origin_dates=prepared.training_origin_dates,
        scenario_origin_dates=prepared.scenario_origin_dates,
        fit_origin_dates=prepared.fit_origin_dates,
        validation_origin_dates=prepared.validation_origin_dates,
        validation_blocks=prepared.validation_date_blocks,
        schedule_family=WEEKLY_RIDGE_SCHEDULE_FAMILY,
        target_labels=tuple(item.claim.target.label for item in prepared.targets),
        target_quantities=tuple(item.claim_quantity for item in prepared.targets),
        objective=objective,
        risk_parameter=risk_parameter,
        validation_fraction=base_config.validation_fraction,
        position_cap=position_cap,
        maximum_latent_coordinate=maximum_latent_coordinate,
        candidates=tuple(candidate_results),
        selected_ridge=selected.ridge,
        selected_by_one_standard_error=True,
        source_commit=source_commit,
        code_version=str(HistoricalSanosConfig.version),
    )


def select_weekly_ridge(
    candidates: tuple[WeeklyRidgeCandidateResult, ...],
) -> WeeklyRidgeCandidateResult:
    """Choose the largest feasible ridge within one SE of the best mean."""
    finite_feasible = tuple(
        candidate
        for candidate in candidates
        if candidate.ridge is not None and candidate.feasible
    )
    if not finite_feasible:
        raise RuntimeError("weekly ridge study has no feasible finite candidate")
    feasible = tuple(candidate for candidate in candidates if candidate.feasible)
    best = min(
        feasible,
        key=lambda value: (
            value.mean_regret,
            math.inf if value.ridge is None else value.ridge,
        ),
    )
    threshold = best.mean_regret + best.standard_error
    eligible = tuple(
        candidate for candidate in feasible if candidate.mean_regret <= threshold
    )
    return max(
        eligible,
        key=lambda value: math.inf if value.ridge is None else value.ridge,
    )


def _null_candidate_result(
    *,
    block_count: int,
    component_count: int,
) -> WeeklyRidgeCandidateResult:
    """Return the exact zero-position benchmark with no fitted parameters."""
    zeros = (0.0,) * block_count
    return WeeklyRidgeCandidateResult(
        ridge=None,
        block_regrets=zeros,
        pricing_block_regrets=zeros,
        zero_block_regrets=zeros,
        execution_block_regrets=zeros,
        mean_regret=0.0,
        standard_error=0.0,
        feasible=True,
        selection_converged_components=component_count,
        final_converged_components=component_count,
        converged_components=2 * component_count,
        component_count=2 * component_count,
        selection_cap_hits=0,
        internal_cap_hits=0,
        executable_cap_hits=0,
        cap_hits=0,
        unregularized_recession_directions=0,
        recession_directions=0,
        maximum_objective_error=0.0,
        maximum_constraint_violation=0.0,
        maximum_latent_coordinate=0.0,
    )


def write_weekly_ridge_selection(
    selection: WeeklyRidgeSelection,
    *,
    output: Path,
) -> None:
    """Atomically write one weekly ridge selection artifact."""
    payload = selection.to_dict()
    payload["selection_sha256"] = selection.sha256
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_weekly_ridge_bundle(
    bundle: WeeklyRidgeBundle,
    *,
    output: Path,
) -> None:
    """Atomically write one complete objective/risk ridge bundle."""
    payload = bundle.to_dict()
    payload["bundle_sha256"] = bundle.sha256
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_weekly_ridge_selection(
    path: Path,
    *,
    expected_sha256: str,
    expected_dataset_manifest_sha256: str,
    expected_target_notional_artifact_sha256: str,
    expected_source_commit: str,
    expected_position_cap: float,
    expected_objective: HistoricalSanosObjective | None = None,
    expected_risk_parameter: float | None = None,
) -> WeeklyRidgeSelection:
    """Load and authenticate one immutable weekly ridge artifact."""
    artifact = Path(path)
    if _file_sha256(artifact) != expected_sha256:
        raise RuntimeError("weekly ridge artifact SHA-256 mismatch")
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("weekly ridge artifact must contain an object")
    declared_sha256 = payload.pop("selection_sha256", None)
    selection = WeeklyRidgeSelection.from_dict(payload)
    if declared_sha256 != selection.sha256:
        raise ValueError("weekly ridge selection digest is invalid")
    if selection.dataset_manifest_sha256 != expected_dataset_manifest_sha256:
        raise ValueError("weekly ridge dataset identity mismatch")
    if (
        selection.target_notional_artifact_sha256
        != expected_target_notional_artifact_sha256
    ):
        raise ValueError("weekly ridge target-notional identity mismatch")
    if selection.code_version != str(HistoricalSanosConfig.version):
        raise ValueError("weekly ridge code version is incompatible")
    if selection.source_commit != expected_source_commit:
        raise ValueError("weekly ridge source commit is incompatible")
    if not math.isclose(
        selection.position_cap,
        expected_position_cap,
        rel_tol=0.0,
        abs_tol=1.0e-14,
    ):
        raise ValueError("weekly ridge position cap is incompatible")
    if expected_objective is not None and selection.objective is not expected_objective:
        raise ValueError("weekly ridge objective identity mismatch")
    if expected_risk_parameter is not None and not math.isclose(
        selection.risk_parameter,
        expected_risk_parameter,
        rel_tol=0.0,
        abs_tol=1.0e-14,
    ):
        raise ValueError("weekly ridge risk-parameter identity mismatch")
    if _file_sha256(artifact) != expected_sha256:
        raise RuntimeError("weekly ridge artifact changed while loading")
    return selection


def load_weekly_ridge_bundle(
    path: Path,
    *,
    expected_sha256: str,
    expected_dataset_manifest_sha256: str,
    expected_target_notional_artifact_sha256: str,
    expected_source_commit: str,
    expected_position_cap: float,
) -> WeeklyRidgeBundle:
    """Load and authenticate a complete objective/risk ridge bundle."""
    artifact = Path(path)
    if _file_sha256(artifact) != expected_sha256:
        raise RuntimeError("weekly ridge bundle SHA-256 mismatch")
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("weekly ridge bundle must contain an object")
    declared_sha256 = payload.pop("bundle_sha256", None)
    bundle = WeeklyRidgeBundle.from_dict(payload)
    if declared_sha256 != bundle.sha256:
        raise ValueError("weekly ridge bundle digest is invalid")
    if bundle.dataset_manifest_sha256 != expected_dataset_manifest_sha256:
        raise ValueError("weekly ridge bundle dataset identity mismatch")
    if (
        bundle.target_notional_artifact_sha256
        != expected_target_notional_artifact_sha256
    ):
        raise ValueError("weekly ridge bundle target-notional identity mismatch")
    if any(
        selection.code_version != str(HistoricalSanosConfig.version)
        for selection in bundle.selections
    ):
        raise ValueError("weekly ridge bundle code version is incompatible")
    if any(
        selection.source_commit != expected_source_commit
        for selection in bundle.selections
    ):
        raise ValueError("weekly ridge bundle source commit is incompatible")
    if any(
        not math.isclose(
            selection.position_cap,
            expected_position_cap,
            rel_tol=0.0,
            abs_tol=1.0e-14,
        )
        for selection in bundle.selections
    ):
        raise ValueError("weekly ridge bundle position cap is incompatible")
    if _file_sha256(artifact) != expected_sha256:
        raise RuntimeError("weekly ridge bundle changed while loading")
    return bundle


def load_weekly_ridge_for_scope(
    path: Path,
    *,
    expected_sha256: str,
    expected_dataset_manifest_sha256: str,
    expected_target_notional_artifact_sha256: str,
    expected_source_commit: str,
    expected_position_cap: float,
    expected_objective: HistoricalSanosObjective,
    expected_risk_parameter: float,
) -> WeeklyRidgeSelection:
    """Load an authenticated selection or select one scope from a bundle."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("weekly ridge artifact must contain an object")
    if payload.get("schema") == WEEKLY_RIDGE_SELECTION_SCHEMA:
        return load_weekly_ridge_selection(
            path,
            expected_sha256=expected_sha256,
            expected_dataset_manifest_sha256=(
                expected_dataset_manifest_sha256
            ),
            expected_target_notional_artifact_sha256=(
                expected_target_notional_artifact_sha256
            ),
            expected_source_commit=expected_source_commit,
            expected_position_cap=expected_position_cap,
            expected_objective=expected_objective,
            expected_risk_parameter=expected_risk_parameter,
        )
    bundle = load_weekly_ridge_bundle(
        path,
        expected_sha256=expected_sha256,
        expected_dataset_manifest_sha256=expected_dataset_manifest_sha256,
        expected_target_notional_artifact_sha256=(
            expected_target_notional_artifact_sha256
        ),
        expected_source_commit=expected_source_commit,
        expected_position_cap=expected_position_cap,
    )
    return bundle.selection_for(
        objective=expected_objective,
        risk_parameter=expected_risk_parameter,
    )


@dataclass(frozen=True, kw_only=True)
class _PreparedTarget:
    claim: WeeklyClaimDefinition
    claim_quantity: float
    scenarios: HistoricalWeeklyScenarioSet


@dataclass(frozen=True, kw_only=True)
class _PreparedOrigin:
    training_origin_dates: tuple[date, ...]
    scenario_origin_dates: tuple[date, ...]
    fit_origin_dates: tuple[date, ...]
    validation_origin_dates: tuple[date, ...]
    validation_indices: NDArray[np.int64]
    validation_blocks: tuple[NDArray[np.int64], ...]
    validation_date_blocks: tuple[tuple[date, ...], ...]
    targets: tuple[_PreparedTarget, ...]


def _prepare_origin(
    panel: PublicHedgingPanel,
    *,
    target_notionals: WeeklyTargetNotionalCalibration,
    origin_date: date,
    validation_fraction: float,
) -> _PreparedOrigin:
    origin_index = panel.dates.index(origin_date)
    scheduled_dates = panel.dates[origin_index : origin_index + 6]
    if len(scheduled_dates) != 6:
        raise ValueError("weekly ridge origin lacks five scheduled sessions")
    training_dates = panel.prior_one_day_origins(origin_date, count=252)
    pairs = tuple(
        (
            panel.one_day_transition(training_date),
            panel.market_day(panel.next_one_session_date(training_date)),
        )
        for training_date in training_dates
    )
    history = HistoricalWeeklyTransitionHistory(
        as_of_date=origin_date,
        transitions=tuple(pair[0] for pair in pairs),
        endpoints=tuple(pair[1] for pair in pairs),
    )
    day = panel.market_day(origin_date)
    quotes = weekly_inception_quotes(
        day,
        terminal_date=scheduled_dates[-1],
        nms_range=(-2.0, 2.0),
        episode_origin_date=origin_date,
    )
    contracts = tuple(quote.contract for quote in quotes)
    claims = build_common_weekly_claims(day)
    quantities = target_notionals.quantity_by_label
    target_labels = tuple(claim.target.label for claim in claims)
    if set(target_labels) != set(quantities):
        raise ValueError("weekly ridge targets do not match target notionals")
    forward_curve = np.asarray(
        [1.0, *(day.forward_at_business_dte(step) for step in range(1, 6))],
        dtype=np.float64,
    )
    model_fit_identity = hashlib.sha256(
        f"{panel.dataset_manifest_sha256}:{origin_date}:weekly-ridge".encode()
    ).hexdigest()
    prepared: list[_PreparedTarget] = []
    reference_scenarios: HistoricalWeeklyScenarioSet | None = None
    for claim in claims:
        quantity = float(quantities[claim.target.label])
        instruction = WeeklyFitInstruction(
            schedule=HedgeSchedule.FROZEN_WEEKLY,
            model="historical-sanos",
            measure="historical-P",
            target_label=claim.target.label,
            claim_quantity=quantity,
            model_config_identity=model_fit_identity,
            policy_config_identity="weekly-ridge-calibration",
            episode_origin_date=origin_date,
            decision_date=origin_date,
            fit_as_of_date=origin_date,
            training_dates=training_dates,
            frozen_contract_ids=tuple(contract.contract_id for contract in contracts),
        )
        state = WeeklyDecisionState(
            instruction=instruction,
            decision_index=0,
            next_session_date=scheduled_dates[1],
            market_day=day,
            contracts=contracts,
            spot_over_episode_origin=1.0,
            inception_forward_over_episode_spot=forward_curve.copy(),
            inception_forward_path=np.ones(1, dtype=np.float64),
            rolling_forward_path=np.ones(1, dtype=np.float64),
            discount_to_origin=1.0,
            option_marks_pv=np.asarray([quote.mark for quote in quotes], dtype=np.float64),
            option_half_spreads_pv=np.asarray(
                [quote.half_spread for quote in quotes], dtype=np.float64
            ),
            option_alive=np.ones(len(quotes), dtype=bool),
            previous_spot_position=0.0,
            previous_option_positions=np.zeros(len(quotes), dtype=np.float64),
        )
        if reference_scenarios is None:
            scenarios = build_historical_weekly_scenarios(
                history,
                state=state,
                claim=claim,
                model_fit_identity=model_fit_identity,
            )
            reference_scenarios = scenarios
        else:
            scenarios = attach_historical_weekly_claim(
                reference_scenarios,
                state=state,
                claim=claim,
            )
        prepared.append(
            _PreparedTarget(
                claim=claim,
                claim_quantity=quantity,
                scenarios=scenarios,
            )
        )
    scenario_count = prepared[0].scenarios.scenarios.scenario_count
    if any(
        item.scenarios.scenarios.scenario_count != scenario_count for item in prepared
    ):
        raise RuntimeError("weekly ridge target scenario counts changed")
    validation_count = max(1, int(round(scenario_count * validation_fraction)))
    validation_indices = np.arange(
        scenario_count - validation_count,
        scenario_count,
        dtype=np.int64,
    )
    scenario_dates = prepared[0].scenarios.training_origin_dates
    split = scenario_count - validation_count
    fit_dates = scenario_dates[:split]
    validation_dates = scenario_dates[split:]
    blocks = tuple(
        np.asarray(block, dtype=np.int64)
        for block in np.array_split(validation_indices, 5)
    )
    date_blocks = tuple(
        tuple(block)
        for block in np.array_split(np.asarray(validation_dates, dtype=object), 5)
    )
    return _PreparedOrigin(
        training_origin_dates=training_dates,
        scenario_origin_dates=scenario_dates,
        fit_origin_dates=fit_dates,
        validation_origin_dates=validation_dates,
        validation_indices=validation_indices,
        validation_blocks=blocks,
        validation_date_blocks=date_blocks,
        targets=tuple(prepared),
    )


def _evaluate_candidate(
    prepared: _PreparedOrigin,
    *,
    config: HistoricalSanosConfig,
    maximum_latent_coordinate: float,
    progress: ProgressFunction | None,
    candidate_index: int,
    candidate_count: int,
    started: float,
) -> WeeklyRidgeCandidateResult:
    context = WeeklyOpenLoopPricingContext(
        prepared.targets[0].scenarios.scenarios,
        config=config,
    )
    fits = []
    pricing_block_regrets: list[FloatArray] = []
    execution_block_regrets: list[FloatArray] = []
    zero_block_regrets: FloatArray | None = None
    for target_index, target in enumerate(prepared.targets, start=1):
        fit = fit_historical_weekly_hedge(
            target.scenarios,
            claim_quantity=target.claim_quantity,
            initial_spot_position=0.0,
            initial_option_positions=np.zeros(
                target.scenarios.scenarios.contract_count,
                dtype=np.float64,
            ),
            config=config,
            pricing_context=context,
        )
        fits.append(fit)
        pricing_policies = (
            (fit.long_policy, target.claim_quantity),
            (fit.short_policy, -target.claim_quantity),
        )
        for policy, signed_quantity in pricing_policies:
            pricing_block_regrets.append(
                _block_regrets(
                    target.scenarios.scenarios,
                    policy=policy,
                    signed_claim_quantity=signed_quantity,
                    blocks=prepared.validation_blocks,
                    config=config,
                )
            )
        execution_block_regrets.append(
            _block_regrets(
                target.scenarios.scenarios,
                policy=fit.pure_policy,
                signed_claim_quantity=target.claim_quantity,
                blocks=prepared.validation_blocks,
                config=config,
            )
        )
        if target_index == 1:
            zero_block_regrets = _block_regrets(
                target.scenarios.scenarios,
                policy=fit.zero_policy,
                signed_claim_quantity=0.0,
                blocks=prepared.validation_blocks,
                config=config,
            )
        _emit(
            progress,
            f"weekly ridge candidate {candidate_index}/{candidate_count} "
            f"ridge={config.quadratic_ridge:g} targets={target_index}/"
            f"{len(prepared.targets)} elapsed_s={time.monotonic() - started:.1f}",
        )
    policies = tuple(
        policy
        for fit in fits
        for policy in (fit.long_policy, fit.short_policy, fit.pure_policy)
    ) + (fits[0].zero_policy,)
    if zero_block_regrets is None:
        raise RuntimeError("weekly ridge calibration lacks its zero-policy stratum")
    pricing_regrets = np.mean(np.stack(pricing_block_regrets), axis=0)
    execution_regrets = np.mean(np.stack(execution_block_regrets), axis=0)
    block_regrets = np.mean(
        np.stack((pricing_regrets, zero_block_regrets, execution_regrets)),
        axis=0,
    )
    standard_error = float(np.std(block_regrets, ddof=1) / math.sqrt(len(block_regrets)))
    maximum_objective_error = max(
        max(policy.selection_objective_error, policy.objective_error)
        for policy in policies
    )
    maximum_constraint_violation = max(
        max(
            policy.selection_maximum_constraint_violation,
            policy.maximum_constraint_violation,
        )
        for policy in policies
    )
    maximum_coordinate = max(
        _maximum_policy_coordinate(policy) for policy in policies
    )
    selection_cap_hits = sum(policy.selection_cap_hits for policy in policies)
    internal_cap_hits = selection_cap_hits + sum(
        policy.cap_hits for policy in policies
    )
    executable_cap_hits = sum(
        _economic_cap_hits(fit.pure_policy, position_cap=config.position_cap)
        for fit in fits
    )
    cap_hits = internal_cap_hits + executable_cap_hits
    unregularized_recession_directions = sum(
        policy.unregularized_recession_direction for policy in policies
    )
    recession_directions = sum(policy.recession_direction for policy in policies)
    selection_converged_components = sum(
        policy.selection_converged for policy in policies
    )
    final_converged_components = sum(policy.converged for policy in policies)
    converged_components = (
        selection_converged_components + final_converged_components
    )
    component_count = 2 * len(policies)
    feasible = bool(
        converged_components == component_count
        and cap_hits == 0
        and recession_directions == 0
        and maximum_objective_error <= config.objective_agreement_tolerance
        and maximum_constraint_violation <= config.certificate_tolerance
        and maximum_coordinate < maximum_latent_coordinate
    )
    return WeeklyRidgeCandidateResult(
        ridge=config.quadratic_ridge,
        block_regrets=tuple(float(value) for value in block_regrets),
        pricing_block_regrets=tuple(float(value) for value in pricing_regrets),
        zero_block_regrets=tuple(
            float(value) for value in zero_block_regrets
        ),
        execution_block_regrets=tuple(
            float(value) for value in execution_regrets
        ),
        mean_regret=float(np.mean(block_regrets)),
        standard_error=standard_error,
        feasible=feasible,
        selection_converged_components=selection_converged_components,
        final_converged_components=final_converged_components,
        converged_components=converged_components,
        component_count=component_count,
        selection_cap_hits=selection_cap_hits,
        internal_cap_hits=internal_cap_hits,
        executable_cap_hits=executable_cap_hits,
        cap_hits=cap_hits,
        unregularized_recession_directions=(
            unregularized_recession_directions
        ),
        recession_directions=recession_directions,
        maximum_objective_error=maximum_objective_error,
        maximum_constraint_violation=maximum_constraint_violation,
        maximum_latent_coordinate=maximum_coordinate,
    )


def _block_regrets(
    scenarios: WeeklyScenarioBatch,
    *,
    policy: HistoricalWeeklyOpenLoopFit,
    signed_claim_quantity: float,
    blocks: tuple[NDArray[np.int64], ...],
    config: HistoricalSanosConfig,
) -> FloatArray:
    regrets = np.empty(len(blocks), dtype=np.float64)
    for block_index, indices in enumerate(blocks):
        subset = _subset_scenarios(scenarios, indices)
        fitted = weekly_scenario_accounting(
            subset,
            spot_positions=policy.selection_spot_positions,
            option_positions=policy.selection_option_positions,
            signed_claim_quantity=signed_claim_quantity,
            initial_spot_position=0.0,
            initial_option_positions=np.zeros(subset.contract_count, dtype=np.float64),
        ).terminal_wealth
        zero = weekly_scenario_accounting(
            subset,
            spot_positions=np.zeros(subset.step_count, dtype=np.float64),
            option_positions=np.zeros(
                (subset.step_count, subset.contract_count), dtype=np.float64
            ),
            signed_claim_quantity=signed_claim_quantity,
            initial_spot_position=0.0,
            initial_option_positions=np.zeros(subset.contract_count, dtype=np.float64),
        ).terminal_wealth
        objective = WeeklyRiskObjective(config.objective.value)
        regrets[block_index] = weekly_certainty_equivalent(
            zero,
            objective=objective,
            risk_parameter=config.risk_parameter,
        ) - weekly_certainty_equivalent(
            fitted,
            objective=objective,
            risk_parameter=config.risk_parameter,
        )
    return regrets


def _subset_scenarios(
    scenarios: WeeklyScenarioBatch,
    indices: NDArray[np.int64],
) -> WeeklyScenarioBatch:
    return replace(
        scenarios,
        inception_forward_paths=scenarios.inception_forward_paths[indices],
        spot_values_pv=scenarios.spot_values_pv[indices],
        spot_step_gains=scenarios.spot_step_gains[indices],
        option_step_gains=scenarios.option_step_gains[indices],
        option_half_spreads_pv=scenarios.option_half_spreads_pv[indices],
        terminal_claim_values_pv=scenarios.terminal_claim_values_pv[indices],
    )


def _maximum_policy_coordinate(policy: HistoricalWeeklyOpenLoopFit) -> float:
    return float(
        max(
            np.max(np.abs(policy.selection_spot_positions)),
            np.max(np.abs(policy.selection_option_positions)),
            np.max(np.abs(policy.spot_positions)),
            np.max(np.abs(policy.option_positions)),
        )
    )


def _economic_cap_hits(
    policy: HistoricalWeeklyOpenLoopFit,
    *,
    position_cap: float,
) -> int:
    selection = np.concatenate(
        (policy.selection_spot_positions[:, None], policy.selection_option_positions),
        axis=1,
    )
    final = np.concatenate(
        (policy.spot_positions[:, None], policy.option_positions),
        axis=1,
    )
    return int(
        np.count_nonzero(np.abs(selection) >= 0.999 * position_cap)
        + np.count_nonzero(np.abs(final) >= 0.999 * position_cap)
    )


def _emit(progress: ProgressFunction | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _canonical_json(payload: dict[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def repository_source_commit(*, require_clean: bool = False) -> str:
    """Return the repository commit containing this module."""
    repository = next(
        (
            parent
            for parent in Path(__file__).resolve().parents
            if (parent / ".git").exists()
        ),
        None,
    )
    if repository is None:
        raise RuntimeError("weekly ridge source repository is unavailable")
    commit = subprocess.check_output(
        ("git", "-C", str(repository), "rev-parse", "HEAD"),
        text=True,
    ).strip()
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise RuntimeError("weekly ridge source commit is invalid")
    if require_clean:
        status = subprocess.check_output(
            (
                "git",
                "-C",
                str(repository),
                "status",
                "--porcelain",
                "--untracked-files=normal",
            ),
            text=True,
        )
        if status.strip():
            raise RuntimeError("weekly ridge calibration requires a clean worktree")
    return commit


def _bundle_design_identity(selection: WeeklyRidgeSelection) -> tuple[object, ...]:
    return (
        selection.origin_date,
        selection.training_origin_dates,
        selection.scenario_origin_dates,
        selection.fit_origin_dates,
        selection.validation_origin_dates,
        selection.validation_blocks,
        selection.schedule_family,
        selection.target_labels,
        selection.target_quantities,
        selection.validation_fraction,
        selection.position_cap,
        selection.maximum_latent_coordinate,
        tuple(candidate.ridge for candidate in selection.candidates),
        selection.source_commit,
        selection.code_version,
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
