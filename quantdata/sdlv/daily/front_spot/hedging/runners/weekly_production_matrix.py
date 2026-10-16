"""Versioned production matrix for the complete weekly hedging benchmark."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from cdxcore.err import verify_inp
from cdxcore.version import version

from quantdata.hedging.protocols import HedgeSchedule
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly import weekly_targets
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    default_public_candidate_path,
)
from quantdata.sdlv.daily.front_spot.hedging.models.checkpointing import file_sha256
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.selection import (
    load_gromm_architecture_selection,
)
from quantdata.sdlv.daily.front_spot.hedging.models.historical_sanos.config import (
    HistoricalSanosObjective,
)
from quantdata.sdlv.daily.front_spot.hedging.models.historical_sanos.weekly_ridge import (
    load_weekly_ridge_bundle,
    repository_source_commit,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.fit_rolling_gromm2_range import (
    load_completed_rolling_gromm2_range,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.generated_evidence import (
    GeneratedProductionGateEvidence,
    GeneratedSelectionEvidence,
    generated_scope_warning_json,
    load_generated_initializer_evidence,
    load_generated_production_gate,
    load_generated_selection_evidence,
    require_generated_production_scope,
    reviewable_repository_commit,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.production_matrix import (
    DailyProductionCase,
    GeneratedProductionInputs,
    GrommProductionInputs,
    ProductionReadiness,
    ProductionResource,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.weekly import (
    WeeklyEvaluationSpec,
)

WEEKLY_INPUT_SCHEMA = "hedging-weekly-input-preflight-v1"
WEEKLY_MATRIX_SCHEMA = "hedging-weekly-production-matrix-v1"


@version("0.2", dependencies=[WeeklyEvaluationSpec])
@dataclass(frozen=True, kw_only=True)
class WeeklyProductionMatrixConfig:
    """Precommitted objective grids and shared weekly run controls."""

    mean_variance_parameters: tuple[float, ...] = (0.1, 1.0, 10.0)
    mean_volatility_parameters: tuple[float, ...] = (0.1, 1.0, 2.0, 3.0)
    entropy_parameters: tuple[float, ...] = (0.1, 1.0, 10.0)
    power_exponential_parameters: tuple[float, ...] = (0.1, 1.0, 10.0)
    observation_count: int = 252
    scenario_count: int = 4_096
    position_cap: float = 1_000.0
    spot_cost_bps: float = 1.0
    evaluation_start: date = date(2023, 1, 1)
    evaluation_end: date = date(2025, 12, 31)

    def __post_init__(self) -> None:
        for values in (
            self.mean_variance_parameters,
            self.mean_volatility_parameters,
            self.entropy_parameters,
            self.power_exponential_parameters,
        ):
            verify_inp(
                bool(values)
                and len(set(values)) == len(values)
                and all(math.isfinite(value) and value > 0.0 for value in values),
                lambda: "weekly risk grids must be unique finite-positive values",
            )
        verify_inp(
            self.observation_count == 252,
            lambda: "the weekly benchmark requires a 252-session rolling window",
        )
        verify_inp(
            self.scenario_count >= 4 and self.scenario_count % 2 == 0,
            lambda: "weekly scenario_count must be even and at least four",
        )
        verify_inp(
            math.isfinite(self.position_cap) and self.position_cap > 0.0,
            lambda: "weekly position_cap must be finite-positive",
        )
        verify_inp(
            math.isfinite(self.spot_cost_bps) and self.spot_cost_bps >= 0.0,
            lambda: "weekly spot_cost_bps must be finite and non-negative",
        )
        verify_inp(
            self.evaluation_start <= self.evaluation_end,
            lambda: "weekly evaluation start must not exceed end",
        )

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible configuration snapshot."""
        payload = asdict(self)
        payload["evaluation_start"] = self.evaluation_start.isoformat()
        payload["evaluation_end"] = self.evaluation_end.isoformat()
        return payload


def build_weekly_production_matrix(
    *,
    dataset: Path,
    output_root: Path,
    checkpoint_root: Path,
    weekly_input_preflight: Path | None,
    historical_weekly_ridge_selection: Path | None = None,
    historical_weekly_ridge_selection_sha256: str | None = None,
    config: WeeklyProductionMatrixConfig = WeeklyProductionMatrixConfig(),
    gromm: GrommProductionInputs = GrommProductionInputs(),
    generated: GeneratedProductionInputs = GeneratedProductionInputs(),
) -> tuple[DailyProductionCase, ...]:
    """Build every precommitted weekly case without launching material work."""
    target_artifact, target_sha256, common_reason = _weekly_input_evidence(
        dataset=dataset,
        preflight_path=weekly_input_preflight,
    )
    historical_reason = _historical_ridge_evidence(
        dataset=dataset,
        target_notional_sha256=target_sha256,
        common_reason=common_reason,
        selection_path=historical_weekly_ridge_selection,
        selection_sha256=historical_weekly_ridge_selection_sha256,
        config=config,
    )
    cases: list[DailyProductionCase] = []
    for schedule in _schedules():
        reference_id = f"{schedule.value}-sanos-lv-reference"
        cases.append(
            _case(
                ready=common_reason is None,
                reason=common_reason,
                case_id=reference_id,
                model="sanos-lv",
                measure="Q-price/historical-transition",
                variant=f"{schedule.value}-reference-delta-eight-factor-vega",
                objective=None,
                risk_parameter=None,
                resource=ProductionResource.CPU,
                command=_weekly_command(
                    model="sanos-lv",
                    schedule=schedule,
                    case_id=reference_id,
                    dataset=dataset,
                    output_root=output_root,
                    checkpoint_root=checkpoint_root,
                    target_notional_artifact=target_artifact,
                    target_notional_sha256=target_sha256,
                    config=config,
                ),
            )
        )
        for objective, parameters in _objective_grids(config):
            for parameter in parameters:
                case_id = _risk_case_id(
                    f"{schedule.value}-historical-sanos",
                    objective,
                    parameter,
                )
                cases.append(
                    _case(
                        ready=historical_reason is None,
                        reason=historical_reason,
                        case_id=case_id,
                        model="historical-sanos",
                        measure="historical-P",
                        variant=f"{schedule.value}-rolling-fixed-contract-scenarios",
                        objective=objective,
                        risk_parameter=parameter,
                        resource=ProductionResource.CPU,
                        command=_weekly_command(
                            model="historical-sanos",
                            schedule=schedule,
                            case_id=case_id,
                            dataset=dataset,
                            output_root=output_root,
                            checkpoint_root=checkpoint_root,
                            target_notional_artifact=target_artifact,
                            target_notional_sha256=target_sha256,
                            config=config,
                            objective=objective,
                            risk_parameter=parameter,
                            historical_weekly_ridge_selection=(
                                historical_weekly_ridge_selection
                            ),
                            historical_weekly_ridge_selection_sha256=(
                                historical_weekly_ridge_selection_sha256
                            ),
                        ),
                    )
                )
    cases.extend(
        _gromm_cases(
            dataset=dataset,
            output_root=output_root,
            checkpoint_root=checkpoint_root,
            target_notional_artifact=target_artifact,
            target_notional_sha256=target_sha256,
            common_reason=common_reason,
            config=config,
            inputs=gromm,
        )
    )
    cases.extend(
        _generated_cases(
            dataset=dataset,
            output_root=output_root,
            checkpoint_root=checkpoint_root,
            target_notional_artifact=target_artifact,
            target_notional_sha256=target_sha256,
            common_reason=common_reason,
            config=config,
            generated=generated,
        )
    )
    identities = tuple(case.case_id for case in cases)
    if len(set(identities)) != len(identities):
        raise RuntimeError("weekly production matrix contains duplicate identities")
    return tuple(cases)


def _weekly_input_evidence(
    *,
    dataset: Path,
    preflight_path: Path | None,
) -> tuple[Path | None, str | None, str | None]:
    if preflight_path is None:
        return None, None, "missing authenticated weekly input preflight"
    try:
        payload = _read_json_object(preflight_path)
        dataset_sha256 = file_sha256(Path(dataset) / "manifest.json")
        if payload.get("schema") != WEEKLY_INPUT_SCHEMA:
            raise ValueError("unexpected weekly input-preflight schema")
        if payload.get("dataset_manifest_sha256") != dataset_sha256:
            raise ValueError("weekly input preflight uses another dataset")
        if payload.get("target_count") != len(weekly_targets()):
            raise ValueError("weekly input preflight does not contain 13 targets")
        if not isinstance(payload.get("episode_count"), int) or payload["episode_count"] <= 0:
            raise ValueError("weekly input preflight has no sizing episodes")
        artifact = Path(_require_string(payload, "target_notional_artifact"))
        target_sha256 = _require_string(
            payload,
            "target_notional_artifact_sha256",
        )
        if file_sha256(artifact) != target_sha256:
            raise ValueError("weekly target-notional artifact hash mismatch")
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as error:
        return None, None, f"invalid weekly input evidence: {error}"
    return artifact, target_sha256, None


def _historical_ridge_evidence(
    *,
    dataset: Path,
    target_notional_sha256: str | None,
    common_reason: str | None,
    selection_path: Path | None,
    selection_sha256: str | None,
    config: WeeklyProductionMatrixConfig,
) -> str | None:
    if common_reason is not None:
        return common_reason
    if selection_path is None or selection_sha256 is None:
        return "missing authenticated Historical-SANOS weekly ridge selection"
    assert target_notional_sha256 is not None
    try:
        bundle = load_weekly_ridge_bundle(
            selection_path,
            expected_sha256=selection_sha256,
            expected_dataset_manifest_sha256=file_sha256(
                Path(dataset) / "manifest.json"
            ),
            expected_target_notional_artifact_sha256=target_notional_sha256,
            expected_source_commit=repository_source_commit(),
            expected_position_cap=config.position_cap,
        )
        for objective, parameters in _objective_grids(config):
            for parameter in parameters:
                bundle.selection_for(
                    objective=HistoricalSanosObjective(objective),
                    risk_parameter=parameter,
                )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        return f"invalid Historical-SANOS weekly ridge evidence: {error}"
    return None


def _gromm_cases(
    *,
    dataset: Path,
    output_root: Path,
    checkpoint_root: Path,
    target_notional_artifact: Path | None,
    target_notional_sha256: str | None,
    common_reason: str | None,
    config: WeeklyProductionMatrixConfig,
    inputs: GrommProductionInputs,
) -> tuple[DailyProductionCase, ...]:
    market_missing = []
    if inputs.market_artifact_root is None:
        market_missing.append("complete rolling market-artifact root")
    if inputs.artifact_range_summary is None:
        market_missing.append("completed rolling fit-range summary")
    range_summary = None
    market_reason = common_reason
    if market_reason is None and market_missing:
        market_reason = "missing " + ", ".join(market_missing)
    if market_reason is None:
        try:
            dataset_sha256 = file_sha256(Path(dataset) / "manifest.json")
            range_summary = load_completed_rolling_gromm2_range(
                inputs.artifact_range_summary,
                dataset=dataset,
                dataset_manifest_sha256=dataset_sha256,
                artifact_root=inputs.market_artifact_root,
            )
            if (
                range_summary.evaluation_start != config.evaluation_start
                or range_summary.evaluation_end != config.evaluation_end
            ):
                raise ValueError("rolling fit range does not match weekly evaluation")
        except (FileNotFoundError, OSError, ValueError) as error:
            market_reason = f"invalid GROMM production evidence: {error}"
    architecture_reason = market_reason
    if architecture_reason is None and inputs.architecture_selection is None:
        architecture_reason = "missing authenticated 2022 hedge-architecture selection"
    if architecture_reason is None:
        try:
            load_gromm_architecture_selection(
                inputs.architecture_selection,
                expected_dataset_manifest_sha256=file_sha256(
                    Path(dataset) / "manifest.json"
                ),
            )
        except (FileNotFoundError, OSError, ValueError) as error:
            architecture_reason = f"invalid GROMM architecture evidence: {error}"
    cases: list[DailyProductionCase] = []
    for schedule in _schedules():
        conventional_id = f"{schedule.value}-gromm-q-dlv-vega"
        conventional = {
            "case_id": conventional_id,
            "model": "gromm-q-dlv-vega",
            "measure": "Q-price/historical-transition",
            "variant": f"{schedule.value}-q-expectation-eight-factor-dlv-vega",
            "objective": None,
            "risk_parameter": None,
            "resource": ProductionResource.GPU,
        }
        if market_reason is not None:
            cases.append(_blocked_case(reason=market_reason, **conventional))
        else:
            cases.append(
                _ready_case(
                    command=_weekly_command(
                        model="gromm-q-dlv-vega",
                        schedule=schedule,
                        case_id=conventional_id,
                        dataset=dataset,
                        output_root=output_root,
                        checkpoint_root=checkpoint_root,
                        target_notional_artifact=target_notional_artifact,
                        target_notional_sha256=target_notional_sha256,
                        config=config,
                        device="cuda",
                        measure="Q",
                        market_artifact_root=inputs.market_artifact_root,
                        market_artifact_range_summary=inputs.artifact_range_summary,
                    ),
                    **conventional,
                )
            )
        for measure in ("P", "Q"):
            for objective, parameters in _objective_grids(config):
                for parameter in parameters:
                    case_id = _risk_case_id(
                        f"{schedule.value}-gromm-{measure.lower()}",
                        objective,
                        parameter,
                    )
                    common = {
                        "case_id": case_id,
                        "model": "gromm",
                        "measure": measure,
                        "variant": (
                            f"{schedule.value}-rolling-16-factor-eight-substep-deep-hedge"
                        ),
                        "objective": objective,
                        "risk_parameter": parameter,
                        "resource": ProductionResource.GPU,
                    }
                    reason = architecture_reason
                    if (
                        reason is None
                        and measure == "P"
                        and range_summary is not None
                        and range_summary.physical_accepted_count
                        != range_summary.eligible_origin_count
                    ):
                        reason = (
                            "rolling physical-drift gate passed for only "
                            f"{range_summary.physical_accepted_count}/"
                            f"{range_summary.eligible_origin_count} origins"
                        )
                    if reason is not None:
                        cases.append(_blocked_case(reason=reason, **common))
                        continue
                    cases.append(
                        _ready_case(
                            command=_weekly_command(
                                model="gromm",
                                schedule=schedule,
                                case_id=case_id,
                                dataset=dataset,
                                output_root=output_root,
                                checkpoint_root=checkpoint_root,
                                target_notional_artifact=target_notional_artifact,
                                target_notional_sha256=target_notional_sha256,
                                config=config,
                                objective=objective,
                                risk_parameter=parameter,
                                device="cuda",
                                measure=measure,
                                market_artifact_root=inputs.market_artifact_root,
                                market_artifact_range_summary=(
                                    inputs.artifact_range_summary
                                ),
                                architecture_selection=inputs.architecture_selection,
                            ),
                            **common,
                        )
                    )
    return tuple(cases)


def _generated_cases(
    *,
    dataset: Path,
    output_root: Path,
    checkpoint_root: Path,
    target_notional_artifact: Path | None,
    target_notional_sha256: str | None,
    common_reason: str | None,
    config: WeeklyProductionMatrixConfig,
    generated: GeneratedProductionInputs,
) -> tuple[DailyProductionCase, ...]:
    dataset_sha256 = _dataset_manifest_sha256(dataset)
    volgan_gate, volgan_gate_reason = _generated_gate(
        generated.volgan_production_gate,
        expected_model="volgan",
        dataset_sha256=dataset_sha256,
        expected_reviewed_commit=generated.reviewed_commit,
    )
    diffusion_gate, diffusion_gate_reason = _generated_gate(
        generated.diffusion_production_gate,
        expected_model="diffusion-ad-seq-vol",
        dataset_sha256=dataset_sha256,
        expected_reviewed_commit=generated.reviewed_commit,
    )
    diffusion_ft_gate, diffusion_ft_gate_reason = _generated_gate(
        generated.diffusion_ft_production_gate,
        expected_model="diffusion-ad-seq-vol-ft",
        dataset_sha256=dataset_sha256,
        expected_reviewed_commit=generated.reviewed_commit,
    )
    volgan_evidence, volgan_reason = _generated_selection(
        generated.volgan_initializer_manifest,
        generated.volgan_selection_source_manifest,
        generated.volgan_epoch_selection,
        expected_model="volgan",
        dataset_sha256=dataset_sha256,
    )
    diffusion_evidence, diffusion_reason = _generated_selection(
        generated.diffusion_initializer_manifest,
        generated.diffusion_selection_source_manifest,
        generated.diffusion_epoch_selection,
        expected_model="diffusion-ad-seq-vol",
        dataset_sha256=dataset_sha256,
    )
    diffusion_base, diffusion_base_reason = _generated_initializer(
        generated.diffusion_initializer_manifest,
        expected_model="diffusion-ad-seq-vol",
        dataset_sha256=dataset_sha256,
    )
    diffusion_ft_evidence, diffusion_ft_reason = _generated_selection(
        generated.diffusion_ft_initializer_manifest,
        generated.diffusion_ft_selection_source_manifest,
        generated.diffusion_ft_epoch_selection,
        expected_model="diffusion-ad-seq-vol-ft",
        dataset_sha256=dataset_sha256,
        base_initializer_manifest=generated.diffusion_initializer_manifest,
        selection_base_initializer_manifest=(
            generated.diffusion_selection_source_manifest
        ),
    )
    cases: list[DailyProductionCase] = []
    for schedule in _schedules():
        cases.extend(
            (
                _generated_case(
                    case_id=f"{schedule.value}-volgan-public-settings",
                    model="volgan",
                    schedule=schedule,
                    variant="published-settings-full-rolling-fit",
                    common_reason=common_reason,
                    gate=volgan_gate,
                    gate_path=generated.volgan_production_gate,
                    gate_reason=volgan_gate_reason,
                    evidence=None,
                    evidence_reason=None,
                    base_evidence=None,
                    rolling=False,
                    dataset=dataset,
                    output_root=output_root,
                    checkpoint_root=checkpoint_root,
                    target_notional_artifact=target_notional_artifact,
                    target_notional_sha256=target_notional_sha256,
                    config=config,
                ),
                _generated_case(
                    case_id=f"{schedule.value}-volgan-rolling-trained",
                    model="volgan",
                    schedule=schedule,
                    variant="fixed-2022-source-selected-update",
                    common_reason=common_reason,
                    gate=volgan_gate,
                    gate_path=generated.volgan_production_gate,
                    gate_reason=volgan_gate_reason,
                    evidence=volgan_evidence,
                    evidence_reason=volgan_reason,
                    base_evidence=None,
                    rolling=True,
                        dataset=dataset,
                        output_root=output_root,
                        checkpoint_root=checkpoint_root,
                        target_notional_artifact=target_notional_artifact,
                        target_notional_sha256=target_notional_sha256,
                    config=config,
                ),
                _generated_case(
                    case_id=f"{schedule.value}-diffusion-ad-seq-vol-public-settings",
                    model="diffusion-ad-seq-vol",
                    schedule=schedule,
                    variant="published-settings-full-rolling-fit",
                    common_reason=common_reason,
                    gate=diffusion_gate,
                    gate_path=generated.diffusion_production_gate,
                    gate_reason=diffusion_gate_reason,
                    evidence=None,
                    evidence_reason=None,
                    base_evidence=None,
                    rolling=False,
                        dataset=dataset,
                        output_root=output_root,
                        checkpoint_root=checkpoint_root,
                        target_notional_artifact=target_notional_artifact,
                        target_notional_sha256=target_notional_sha256,
                    config=config,
                ),
                _generated_case(
                    case_id=f"{schedule.value}-diffusion-ad-seq-vol-rolling-trained",
                    model="diffusion-ad-seq-vol",
                    schedule=schedule,
                    variant="fixed-2022-source-selected-update",
                    common_reason=common_reason,
                    gate=diffusion_gate,
                    gate_path=generated.diffusion_production_gate,
                    gate_reason=diffusion_gate_reason,
                    evidence=diffusion_evidence,
                    evidence_reason=diffusion_reason,
                    base_evidence=None,
                    rolling=True,
                        dataset=dataset,
                        output_root=output_root,
                        checkpoint_root=checkpoint_root,
                        target_notional_artifact=target_notional_artifact,
                        target_notional_sha256=target_notional_sha256,
                    config=config,
                ),
                _generated_case(
                    case_id=f"{schedule.value}-diffusion-ad-seq-vol-ft-public-settings",
                    model="diffusion-ad-seq-vol-ft",
                    schedule=schedule,
                    variant="published-settings-full-rolling-fit",
                    common_reason=common_reason or diffusion_base_reason,
                    gate=diffusion_ft_gate,
                    gate_path=generated.diffusion_ft_production_gate,
                    gate_reason=diffusion_ft_gate_reason,
                    evidence=None,
                    evidence_reason=None,
                    base_evidence=diffusion_base,
                    rolling=False,
                        dataset=dataset,
                        output_root=output_root,
                        checkpoint_root=checkpoint_root,
                        target_notional_artifact=target_notional_artifact,
                        target_notional_sha256=target_notional_sha256,
                    config=config,
                ),
                _generated_case(
                    case_id=f"{schedule.value}-diffusion-ad-seq-vol-ft-rolling-trained",
                    model="diffusion-ad-seq-vol-ft",
                    schedule=schedule,
                    variant="fixed-2022-source-selected-update",
                    common_reason=common_reason,
                    gate=diffusion_ft_gate,
                    gate_path=generated.diffusion_ft_production_gate,
                    gate_reason=diffusion_ft_gate_reason,
                    evidence=diffusion_ft_evidence,
                    evidence_reason=diffusion_ft_reason,
                    base_evidence=diffusion_base,
                    rolling=True,
                        dataset=dataset,
                        output_root=output_root,
                        checkpoint_root=checkpoint_root,
                        target_notional_artifact=target_notional_artifact,
                        target_notional_sha256=target_notional_sha256,
                    config=config,
                ),
            )
        )
    return tuple(cases)


def _generated_case(
    *,
    case_id: str,
    model: str,
    schedule: HedgeSchedule,
    variant: str,
    common_reason: str | None,
    gate: GeneratedProductionGateEvidence | None,
    gate_path: Path | None,
    gate_reason: str | None,
    evidence: GeneratedSelectionEvidence | None,
    evidence_reason: str | None,
    base_evidence: GeneratedSelectionEvidence | None,
    rolling: bool,
    dataset: Path,
    output_root: Path,
    checkpoint_root: Path,
    target_notional_artifact: Path | None,
    target_notional_sha256: str | None,
    config: WeeklyProductionMatrixConfig,
) -> DailyProductionCase:
    reason = common_reason
    if reason is None and rolling and evidence is None:
        reason = evidence_reason or "missing authenticated generated evidence"
    if reason is None and model.endswith("-ft") and base_evidence is None:
        reason = "missing authenticated fine-tuning base initializer"
    common = {
        "case_id": case_id,
        "model": model,
        "measure": "historical-P",
        "variant": f"{schedule.value}-{variant}",
        "objective": "mean-variance",
        "risk_parameter": 1.0,
        "resource": ProductionResource.GPU,
    }
    if reason is not None:
        return _blocked_case(reason=reason, **common)
    command = _weekly_command(
        model=model,
        schedule=schedule,
        case_id=case_id,
        dataset=dataset,
        output_root=output_root,
        checkpoint_root=checkpoint_root,
        target_notional_artifact=target_notional_artifact,
        target_notional_sha256=target_notional_sha256,
        config=config,
        device="cuda",
        objective="mean-variance",
        risk_parameter=1.0,
        epochs=(evidence.selected_epochs if rolling and evidence else None),
        model_label=(
            evidence.initializer_model_label if rolling and evidence else None
        ),
        warm_start_checkpoint=(
            evidence.initializer_checkpoint if rolling and evidence else None
        ),
        warm_start_checkpoint_sha256=(
            evidence.initializer_checkpoint_sha256
            if rolling and evidence
            else None
        ),
        base_checkpoint=(
            base_evidence.initializer_checkpoint if base_evidence else None
        ),
        base_checkpoint_sha256=(
            base_evidence.initializer_checkpoint_sha256
            if base_evidence
            else None
        ),
    )
    if gate is not None:
        scope = f"{schedule.value}-{'rolling' if rolling else 'public'}"
        try:
            require_generated_production_scope(
                gate,
                scope=scope,
                selection=evidence if rolling else None,
                base_initializer=base_evidence,
            )
        except ValueError as error:
            return _blocked_case(
                reason=f"invalid generated production gate: {error}",
                **common,
            )
        return _ready_case(
            command=command,
            reviewed_commit=gate.reviewed_commit,
            production_gate_sha256=gate.gate_sha256,
            production_gate_path=str(Path(gate_path).resolve()),
            production_gate_status=gate.status,
            production_gate_warning_json=generated_scope_warning_json(
                gate,
                scope=scope,
            ),
            **common,
        )
    if gate_path is not None:
        return _blocked_case(
            reason=gate_reason or "invalid generated-model production gate",
            **common,
        )
    return _waiting_gate_case(
        reason=gate_reason or "awaiting generated-model production gate",
        command=command,
        **common,
    )


def _dataset_manifest_sha256(dataset: Path) -> str | None:
    try:
        return file_sha256(Path(dataset) / "manifest.json")
    except (FileNotFoundError, OSError):
        return None


def _generated_gate(
    gate_path: Path | None,
    *,
    expected_model: str,
    dataset_sha256: str | None,
    expected_reviewed_commit: str | None,
) -> tuple[GeneratedProductionGateEvidence | None, str | None]:
    if gate_path is None:
        return None, "awaiting generated-model review and GPU-readiness gate"
    if dataset_sha256 is None:
        return None, "invalid generated production gate: missing public dataset manifest"
    try:
        reviewed_commit = (
            reviewable_repository_commit()
            if expected_reviewed_commit is None
            else expected_reviewed_commit
        )
        return (
            load_generated_production_gate(
                gate_path=gate_path,
                expected_model=expected_model,
                expected_dataset_manifest_sha256=dataset_sha256,
                expected_reviewed_commit=reviewed_commit,
            ),
            None,
        )
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as error:
        return None, f"invalid generated production gate: {error}"


def _generated_selection(
    initializer_manifest: Path | None,
    selection_source_initializer_manifest: Path | None,
    epoch_selection: Path | None,
    *,
    expected_model: str,
    dataset_sha256: str | None,
    base_initializer_manifest: Path | None = None,
    selection_base_initializer_manifest: Path | None = None,
) -> tuple[GeneratedSelectionEvidence | None, str | None]:
    missing = []
    if initializer_manifest is None:
        missing.append("authenticated pre-evaluation initializer manifest")
    if selection_source_initializer_manifest is None:
        missing.append("authenticated 2022 selection-source manifest")
    if epoch_selection is None:
        missing.append("authenticated 2022 epoch selection")
    if dataset_sha256 is None:
        missing.append("public dataset manifest")
    if missing:
        return None, "missing " + ", ".join(missing)
    try:
        return (
            load_generated_selection_evidence(
                initializer_manifest=initializer_manifest,
                selection_source_initializer_manifest=(
                    selection_source_initializer_manifest
                ),
                epoch_selection=epoch_selection,
                expected_model=expected_model,
                expected_dataset_manifest_sha256=dataset_sha256,
                base_initializer_manifest=base_initializer_manifest,
                selection_base_initializer_manifest=(
                    selection_base_initializer_manifest
                ),
            ),
            None,
        )
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as error:
        return None, f"invalid generated production evidence: {error}"


def _generated_initializer(
    initializer_manifest: Path | None,
    *,
    expected_model: str,
    dataset_sha256: str | None,
) -> tuple[GeneratedSelectionEvidence | None, str | None]:
    if initializer_manifest is None or dataset_sha256 is None:
        return None, "missing authenticated fine-tuning base initializer"
    try:
        return (
            load_generated_initializer_evidence(
                initializer_manifest=initializer_manifest,
                expected_model=expected_model,
                expected_dataset_manifest_sha256=dataset_sha256,
            ),
            None,
        )
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as error:
        return None, f"invalid generated base evidence: {error}"


def _weekly_command(
    *,
    model: str,
    schedule: HedgeSchedule,
    case_id: str,
    dataset: Path,
    output_root: Path,
    checkpoint_root: Path,
    target_notional_artifact: Path | None,
    target_notional_sha256: str | None,
    config: WeeklyProductionMatrixConfig,
    objective: str | None = None,
    risk_parameter: float | None = None,
    device: str = "cpu",
    epochs: int | None = None,
    model_label: str | None = None,
    warm_start_checkpoint: Path | None = None,
    warm_start_checkpoint_sha256: str | None = None,
    base_checkpoint: Path | None = None,
    base_checkpoint_sha256: str | None = None,
    measure: str | None = None,
    market_artifact_root: Path | None = None,
    market_artifact_range_summary: Path | None = None,
    architecture_selection: Path | None = None,
    historical_weekly_ridge_selection: Path | None = None,
    historical_weekly_ridge_selection_sha256: str | None = None,
) -> tuple[str, ...]:
    command = [
        "python",
        "-m",
        "quantdata.sdlv.daily.front_spot.hedging.runners.run_weekly",
        "--model",
        model,
        "--schedule",
        schedule.value,
        "--dataset",
        str(dataset),
        "--output",
        str(output_root / case_id),
        "--checkpoint-directory",
        str(checkpoint_root / case_id),
        "--target-notional-artifact",
        "" if target_notional_artifact is None else str(target_notional_artifact),
        "--target-notional-artifact-sha256",
        "" if target_notional_sha256 is None else target_notional_sha256,
        "--observation-count",
        str(config.observation_count),
        "--position-cap",
        f"{config.position_cap:g}",
        "--spot-cost-bps",
        f"{config.spot_cost_bps:g}",
        "--scenario-count",
        str(config.scenario_count),
        "--device",
        device,
        "--start",
        config.evaluation_start.isoformat(),
        "--end",
        config.evaluation_end.isoformat(),
    ]
    dataset_sha256 = _dataset_manifest_sha256(dataset)
    if dataset_sha256 is not None:
        command.extend(("--dataset-manifest-sha256", dataset_sha256))
    if objective is not None:
        command.extend(("--objective", objective))
    if risk_parameter is not None:
        command.extend(("--risk-parameter", f"{risk_parameter:g}"))
    if epochs is not None:
        command.extend(("--epochs", str(epochs)))
    if model_label is not None:
        command.extend(("--model-label", model_label))
    if warm_start_checkpoint is not None:
        command.extend(("--warm-start-checkpoint", str(warm_start_checkpoint)))
    if warm_start_checkpoint_sha256 is not None:
        command.extend(
            ("--warm-start-checkpoint-sha256", warm_start_checkpoint_sha256)
        )
    if base_checkpoint is not None:
        command.extend(("--base-checkpoint", str(base_checkpoint)))
    if base_checkpoint_sha256 is not None:
        command.extend(("--base-checkpoint-sha256", base_checkpoint_sha256))
    if measure is not None:
        command.extend(("--measure", measure))
    if market_artifact_root is not None:
        command.extend(("--market-artifact-root", str(market_artifact_root)))
    if market_artifact_range_summary is not None:
        command.extend(
            ("--market-artifact-range-summary", str(market_artifact_range_summary))
        )
    if architecture_selection is not None:
        command.extend(("--architecture-selection", str(architecture_selection)))
    if historical_weekly_ridge_selection is not None:
        command.extend(
            (
                "--historical-weekly-ridge-selection",
                str(historical_weekly_ridge_selection),
            )
        )
    if historical_weekly_ridge_selection_sha256 is not None:
        command.extend(
            (
                "--historical-weekly-ridge-selection-sha256",
                historical_weekly_ridge_selection_sha256,
            )
        )
    return tuple(command)


def _case(
    *,
    ready: bool,
    reason: str | None,
    command: tuple[str, ...],
    **kwargs: object,
) -> DailyProductionCase:
    if ready:
        return _ready_case(command=command, **kwargs)
    if reason is None:
        raise RuntimeError("blocked weekly case requires a reason")
    return _blocked_case(reason=reason, **kwargs)


def _objective_grids(
    config: WeeklyProductionMatrixConfig,
) -> tuple[tuple[str, tuple[float, ...]], ...]:
    return (
        ("mean-variance", config.mean_variance_parameters),
        ("mean-volatility", config.mean_volatility_parameters),
        ("entropy", config.entropy_parameters),
        ("power-exponential", config.power_exponential_parameters),
    )


def _schedules() -> tuple[HedgeSchedule, ...]:
    return (HedgeSchedule.FROZEN_WEEKLY, HedgeSchedule.RESTARTED_WEEKLY)


def _risk_case_id(prefix: str, objective: str, parameter: float) -> str:
    return f"{prefix}-{objective}-{parameter:g}".replace(".", "p")


def _artifact_identity(path: Path | None) -> dict[str, str | None] | None:
    if path is None:
        return None
    try:
        sha256 = file_sha256(path)
    except (FileNotFoundError, OSError):
        sha256 = None
    return {"path": str(path), "sha256": sha256}


def _ready_case(**kwargs: object) -> DailyProductionCase:
    return DailyProductionCase(
        readiness=ProductionReadiness.READY,
        blocked_reason=None,
        **kwargs,
    )


def _blocked_case(*, reason: str, **kwargs: object) -> DailyProductionCase:
    return DailyProductionCase(
        readiness=ProductionReadiness.BLOCKED,
        blocked_reason=reason,
        command=None,
        **kwargs,
    )


def _waiting_gate_case(
    *,
    reason: str,
    command: tuple[str, ...],
    **kwargs: object,
) -> DailyProductionCase:
    return DailyProductionCase(
        readiness=ProductionReadiness.WAITING_GATE,
        blocked_reason=reason,
        command=command,
        **kwargs,
    )


def _read_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _require_string(payload: dict[str, object], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def main() -> None:
    """Print or save the immutable weekly production matrix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=default_public_candidate_path())
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--weekly-input-preflight", type=Path)
    parser.add_argument("--historical-weekly-ridge-selection", type=Path)
    parser.add_argument("--historical-weekly-ridge-selection-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gromm-market-artifact-root", type=Path)
    parser.add_argument("--gromm-architecture-selection", type=Path)
    parser.add_argument("--gromm-artifact-range-summary", type=Path)
    parser.add_argument("--generated-reviewed-commit")
    parser.add_argument("--volgan-production-gate", type=Path)
    parser.add_argument("--volgan-initializer-manifest", type=Path)
    parser.add_argument("--volgan-selection-source-manifest", type=Path)
    parser.add_argument("--volgan-epoch-selection", type=Path)
    parser.add_argument("--diffusion-production-gate", type=Path)
    parser.add_argument("--diffusion-initializer-manifest", type=Path)
    parser.add_argument("--diffusion-selection-source-manifest", type=Path)
    parser.add_argument("--diffusion-epoch-selection", type=Path)
    parser.add_argument("--diffusion-ft-production-gate", type=Path)
    parser.add_argument("--diffusion-ft-initializer-manifest", type=Path)
    parser.add_argument("--diffusion-ft-selection-source-manifest", type=Path)
    parser.add_argument("--diffusion-ft-epoch-selection", type=Path)
    arguments = parser.parse_args()
    config = WeeklyProductionMatrixConfig()
    cases = build_weekly_production_matrix(
        dataset=arguments.dataset,
        output_root=arguments.output_root,
        checkpoint_root=arguments.checkpoint_root,
        weekly_input_preflight=arguments.weekly_input_preflight,
        historical_weekly_ridge_selection=(
            arguments.historical_weekly_ridge_selection
        ),
        historical_weekly_ridge_selection_sha256=(
            arguments.historical_weekly_ridge_selection_sha256
        ),
        config=config,
        gromm=GrommProductionInputs(
            market_artifact_root=arguments.gromm_market_artifact_root,
            architecture_selection=arguments.gromm_architecture_selection,
            artifact_range_summary=arguments.gromm_artifact_range_summary,
        ),
        generated=GeneratedProductionInputs(
            reviewed_commit=arguments.generated_reviewed_commit,
            volgan_production_gate=arguments.volgan_production_gate,
            volgan_initializer_manifest=arguments.volgan_initializer_manifest,
            volgan_selection_source_manifest=(
                arguments.volgan_selection_source_manifest
            ),
            volgan_epoch_selection=arguments.volgan_epoch_selection,
            diffusion_production_gate=arguments.diffusion_production_gate,
            diffusion_initializer_manifest=(
                arguments.diffusion_initializer_manifest
            ),
            diffusion_selection_source_manifest=(
                arguments.diffusion_selection_source_manifest
            ),
            diffusion_epoch_selection=arguments.diffusion_epoch_selection,
            diffusion_ft_production_gate=(
                arguments.diffusion_ft_production_gate
            ),
            diffusion_ft_initializer_manifest=(
                arguments.diffusion_ft_initializer_manifest
            ),
            diffusion_ft_selection_source_manifest=(
                arguments.diffusion_ft_selection_source_manifest
            ),
            diffusion_ft_epoch_selection=(
                arguments.diffusion_ft_epoch_selection
            ),
        ),
    )
    payload = {
        # The v1 matrix contract permits additive informational fields; queue
        # readers select required fields and ignore authenticated audit metadata.
        "schema": WEEKLY_MATRIX_SCHEMA,
        "code_version": str(type(config).version),
        "config": config.to_dict(),
        "generated_reviewed_commit": arguments.generated_reviewed_commit,
        "weekly_input_preflight": (
            None
            if arguments.weekly_input_preflight is None
            else str(arguments.weekly_input_preflight)
        ),
        "historical_weekly_ridge_selection": _artifact_identity(
            arguments.historical_weekly_ridge_selection
        ),
        "generated_production_gates": {
            "volgan": _artifact_identity(arguments.volgan_production_gate),
            "diffusion-ad-seq-vol": _artifact_identity(
                arguments.diffusion_production_gate
            ),
            "diffusion-ad-seq-vol-ft": _artifact_identity(
                arguments.diffusion_ft_production_gate
            ),
        },
        "case_count": len(cases),
        "ready_count": sum(
            case.readiness is ProductionReadiness.READY for case in cases
        ),
        "waiting_gate_count": sum(
            case.readiness is ProductionReadiness.WAITING_GATE for case in cases
        ),
        "blocked_count": sum(
            case.readiness is ProductionReadiness.BLOCKED for case in cases
        ),
        "economic_cap_warning_case_count": sum(
            case.production_gate_warning_json is not None for case in cases
        ),
        "cases": [case.to_dict() for case in cases],
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(encoded, end="", flush=True)
        return
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(arguments.output)
    print(arguments.output, flush=True)


if __name__ == "__main__":
    main()
