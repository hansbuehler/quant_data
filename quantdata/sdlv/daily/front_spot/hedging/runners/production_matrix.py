"""Versioned production matrix for the complete daily hedging benchmark."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path

from cdxcore.err import verify_inp
from cdxcore.version import version

from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    default_public_candidate_path,
)
from quantdata.sdlv.daily.front_spot.hedging.models.checkpointing import file_sha256
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.selection import (
    load_gromm_architecture_selection,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.daily_results import (
    DailyResultsConfig,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.fit_rolling_gromm2_range import (
    load_completed_rolling_gromm2_range,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.generated_evidence import (
    GeneratedProductionGateEvidence,
    GeneratedSelectionEvidence,
    load_generated_initializer_evidence,
    load_generated_production_gate,
    load_generated_selection_evidence,
    require_generated_production_scope,
    reviewable_repository_commit,
)


class ProductionResource(StrEnum):
    """Primary execution resource for one production case."""

    CPU = "cpu"
    GPU = "gpu"


class ProductionReadiness(StrEnum):
    """Whether a case can be launched by the current daily runner."""

    READY = "ready"
    WAITING_GATE = "waiting-gate"
    BLOCKED = "blocked"


@version("1.3", dependencies=[DailyResultsConfig])
@dataclass(frozen=True, kw_only=True)
class DailyProductionMatrixConfig:
    """Precommitted objective grids and shared daily run controls."""

    mean_variance_parameters: tuple[float, ...] = (0.1, 1.0, 10.0)
    mean_volatility_parameters: tuple[float, ...] = (0.1, 1.0, 2.0, 3.0)
    entropy_parameters: tuple[float, ...] = (0.1, 1.0, 10.0)
    power_exponential_parameters: tuple[float, ...] = (0.1, 1.0, 10.0)
    observation_count: int = 252
    scenario_count: int = 4_096
    position_cap: float = 1_000.0
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
                lambda: "production risk grids must be unique finite-positive values",
            )
        verify_inp(
            self.observation_count == 252,
            lambda: "the benchmark requires a 252-session rolling window",
        )
        verify_inp(
            self.scenario_count >= 4 and self.scenario_count % 2 == 0,
            lambda: "scenario_count must be even and at least four",
        )
        verify_inp(
            math.isfinite(self.position_cap) and self.position_cap > 0.0,
            lambda: "position_cap must be finite-positive",
        )
        verify_inp(
            self.evaluation_start <= self.evaluation_end,
            lambda: "production evaluation start must not exceed end",
        )

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible configuration snapshot."""
        payload = asdict(self)
        payload["evaluation_start"] = self.evaluation_start.isoformat()
        payload["evaluation_end"] = self.evaluation_end.isoformat()
        return payload


@dataclass(frozen=True, kw_only=True)
class GrommProductionInputs:
    """Authenticated artifacts required to enable rolling GROMM cases."""

    market_artifact_root: Path | None = None
    architecture_selection: Path | None = None
    artifact_range_summary: Path | None = None


@dataclass(frozen=True, kw_only=True)
class GeneratedProductionInputs:
    """Authenticated generated-model gates, selections, and reviewed lineage."""

    reviewed_commit: str | None = None
    volgan_production_gate: Path | None = None
    volgan_initializer_manifest: Path | None = None
    volgan_selection_source_manifest: Path | None = None
    volgan_epoch_selection: Path | None = None
    diffusion_production_gate: Path | None = None
    diffusion_initializer_manifest: Path | None = None
    diffusion_selection_source_manifest: Path | None = None
    diffusion_epoch_selection: Path | None = None
    diffusion_ft_production_gate: Path | None = None
    diffusion_ft_initializer_manifest: Path | None = None
    diffusion_ft_selection_source_manifest: Path | None = None
    diffusion_ft_epoch_selection: Path | None = None


@dataclass(frozen=True, kw_only=True)
class DailyProductionCase:
    """One auditable model/objective/resource cell in the benchmark matrix."""

    case_id: str
    model: str
    measure: str
    variant: str
    objective: str | None
    risk_parameter: float | None
    resource: ProductionResource
    readiness: ProductionReadiness
    blocked_reason: str | None
    command: tuple[str, ...] | None
    reviewed_commit: str | None = None
    production_gate_sha256: str | None = None
    production_gate_path: str | None = None

    def __post_init__(self) -> None:
        verify_inp(bool(self.case_id.strip()), lambda: "case_id must be non-empty")
        verify_inp(bool(self.model.strip()), lambda: "model must be non-empty")
        verify_inp(bool(self.measure.strip()), lambda: "measure must be non-empty")
        verify_inp(bool(self.variant.strip()), lambda: "variant must be non-empty")
        verify_inp(
            self.risk_parameter is None
            or (math.isfinite(self.risk_parameter) and self.risk_parameter > 0.0),
            lambda: "risk_parameter must be finite-positive when supplied",
        )
        verify_inp(
            self.reviewed_commit is None
            or (
                len(self.reviewed_commit) == 40
                and all(
                    character in "0123456789abcdef"
                    for character in self.reviewed_commit.lower()
                )
            ),
            lambda: "reviewed_commit must be a Git SHA-1 when supplied",
        )
        verify_inp(
            self.production_gate_sha256 is None
            or (
                len(self.production_gate_sha256) == 64
                and all(
                    character in "0123456789abcdef"
                    for character in self.production_gate_sha256.lower()
                )
            ),
            lambda: "production_gate_sha256 must be a SHA-256 when supplied",
        )
        verify_inp(
            self.production_gate_path is None or bool(self.production_gate_path),
            lambda: "production_gate_path must be non-empty when supplied",
        )
        if self.readiness is ProductionReadiness.READY:
            verify_inp(self.command is not None, lambda: "ready case requires a command")
            verify_inp(
                self.blocked_reason is None,
                lambda: "ready case cannot have a blocked reason",
            )
        elif self.readiness is ProductionReadiness.WAITING_GATE:
            verify_inp(
                self.command is not None,
                lambda: "gate-waiting case requires a staged command",
            )
            verify_inp(
                bool(self.blocked_reason),
                lambda: "gate-waiting case requires a gate description",
            )
        else:
            verify_inp(
                bool(self.blocked_reason),
                lambda: "blocked case requires a reason",
            )
            verify_inp(self.command is None, lambda: "blocked case cannot have a command")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        payload = asdict(self)
        payload["resource"] = self.resource.value
        payload["readiness"] = self.readiness.value
        payload["command"] = None if self.command is None else list(self.command)
        return payload


def build_daily_production_matrix(
    *,
    dataset: Path,
    output_root: Path,
    checkpoint_root: Path,
    config: DailyProductionMatrixConfig = DailyProductionMatrixConfig(),
    gromm: GrommProductionInputs = GrommProductionInputs(),
    generated: GeneratedProductionInputs = GeneratedProductionInputs(),
) -> tuple[DailyProductionCase, ...]:
    """Build every precommitted daily case without launching material work."""
    cases: list[DailyProductionCase] = []
    cases.append(
        _ready_case(
            case_id="sanos-lv-reference",
            model="sanos-lv",
            measure="Q-price/historical-transition",
            variant="reference-delta-eight-factor-vega",
            objective=None,
            risk_parameter=None,
            resource=ProductionResource.CPU,
            command=_daily_command(
                model="sanos-lv",
                case_id="sanos-lv-reference",
                dataset=dataset,
                output_root=output_root,
                checkpoint_root=checkpoint_root,
                config=config,
            ),
        )
    )
    for objective, parameters in (
        ("mean-variance", config.mean_variance_parameters),
        ("mean-volatility", config.mean_volatility_parameters),
        ("power-exponential", config.power_exponential_parameters),
    ):
        for parameter in parameters:
            case_id = _risk_case_id("covariance", objective, parameter)
            cases.append(
                _ready_case(
                    case_id=case_id,
                    model="robust-covariance",
                    measure="historical-P",
                    variant="independent-shrinkage-tyler",
                    objective=objective,
                    risk_parameter=parameter,
                    resource=ProductionResource.CPU,
                    command=_daily_command(
                        model="covariance",
                        case_id=case_id,
                        dataset=dataset,
                        output_root=output_root,
                        checkpoint_root=checkpoint_root,
                        config=config,
                        objective=objective,
                        risk_parameter=parameter,
                    ),
                )
            )
    for objective, parameters in _objective_grids(config):
        for parameter in parameters:
            case_id = _risk_case_id("historical-sanos", objective, parameter)
            cases.append(
                _ready_case(
                    case_id=case_id,
                    model="historical-sanos",
                    measure="historical-P",
                    variant="rolling-fixed-contract-scenarios",
                    objective=objective,
                    risk_parameter=parameter,
                    resource=ProductionResource.CPU,
                    command=_daily_command(
                        model="historical-sanos",
                        case_id=case_id,
                        dataset=dataset,
                        output_root=output_root,
                        checkpoint_root=checkpoint_root,
                        config=config,
                        objective=objective,
                        risk_parameter=parameter,
                    ),
                )
            )
    cases.extend(
        _gromm_cases(
            dataset=dataset,
            output_root=output_root,
            checkpoint_root=checkpoint_root,
            config=config,
            inputs=gromm,
        )
    )
    cases.extend(
        _generated_cases(
            dataset=dataset,
            output_root=output_root,
            checkpoint_root=checkpoint_root,
            config=config,
            generated=generated,
        )
    )
    identities = tuple(case.case_id for case in cases)
    if len(set(identities)) != len(identities):
        raise RuntimeError("daily production matrix contains duplicate case identities")
    return tuple(cases)


def _gromm_cases(
    *,
    dataset: Path,
    output_root: Path,
    checkpoint_root: Path,
    config: DailyProductionMatrixConfig,
    inputs: GrommProductionInputs,
) -> tuple[DailyProductionCase, ...]:
    market_missing = []
    if inputs.market_artifact_root is None:
        market_missing.append("complete rolling market-artifact root")
    if inputs.artifact_range_summary is None:
        market_missing.append("completed rolling fit-range summary")
    range_summary = None
    market_reason = None
    if not market_missing:
        try:
            dataset_manifest_sha256 = file_sha256(Path(dataset) / "manifest.json")
            range_summary = load_completed_rolling_gromm2_range(
                inputs.artifact_range_summary,
                dataset=dataset,
                dataset_manifest_sha256=dataset_manifest_sha256,
                artifact_root=inputs.market_artifact_root,
            )
            if (
                range_summary.evaluation_start != config.evaluation_start
                or range_summary.evaluation_end != config.evaluation_end
            ):
                raise ValueError(
                    "rolling fit range does not match the production evaluation range"
                )
        except (FileNotFoundError, OSError, ValueError) as error:
            market_reason = f"invalid GROMM production evidence: {error}"
    else:
        market_reason = "missing " + ", ".join(market_missing)
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
    conventional = {
        "case_id": "gromm-q-dlv-vega",
        "model": "gromm-q-dlv-vega",
        "measure": "Q-price/historical-transition",
        "variant": "q-expectation-eight-factor-dlv-vega",
        "objective": None,
        "risk_parameter": None,
        "resource": ProductionResource.CPU,
    }
    cases: list[DailyProductionCase] = []
    if market_reason is not None:
        cases.append(_blocked_case(reason=market_reason, **conventional))
    else:
        cases.append(
            _ready_case(
                command=_daily_command(
                    model="gromm-q-dlv-vega",
                    case_id="gromm-q-dlv-vega",
                    dataset=dataset,
                    output_root=output_root,
                    checkpoint_root=checkpoint_root,
                    config=config,
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
                    f"gromm-{measure.lower()}", objective, parameter
                )
                common = {
                    "case_id": case_id,
                    "model": "gromm",
                    "measure": measure,
                    "variant": "rolling-16-factor-eight-substep-deep-hedge",
                    "objective": objective,
                    "risk_parameter": parameter,
                    "resource": ProductionResource.GPU,
                }
                if architecture_reason is not None:
                    cases.append(_blocked_case(reason=architecture_reason, **common))
                    continue
                assert range_summary is not None
                if (
                    measure == "P"
                    and range_summary.physical_accepted_count
                    != range_summary.eligible_origin_count
                ):
                    cases.append(
                        _blocked_case(
                            reason=(
                                "rolling physical-drift gate passed for only "
                                f"{range_summary.physical_accepted_count}/"
                                f"{range_summary.eligible_origin_count} origins"
                            ),
                            **common,
                        )
                    )
                    continue
                cases.append(
                    _ready_case(
                        command=_daily_command(
                            model="gromm",
                            case_id=case_id,
                            dataset=dataset,
                            output_root=output_root,
                            checkpoint_root=checkpoint_root,
                            config=config,
                            objective=objective,
                            risk_parameter=parameter,
                            device="cuda",
                            measure=measure,
                            market_artifact_root=inputs.market_artifact_root,
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
    config: DailyProductionMatrixConfig,
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
    cases.append(
        _review_gated_case(
            gate=volgan_gate,
            gate_path=generated.volgan_production_gate,
            gate_reason=volgan_gate_reason,
            scope="daily-public",
            case_id="volgan-public-settings",
            model="volgan",
            measure="historical-P",
            variant="published-settings-full-rolling-fit",
            objective="mean-variance",
            risk_parameter=1.0,
            resource=ProductionResource.GPU,
            command=_daily_command(
                model="volgan",
                case_id="volgan-public-settings",
                dataset=dataset,
                output_root=output_root,
                checkpoint_root=checkpoint_root,
                config=config,
                device="cuda",
                objective="mean-variance",
                risk_parameter=1.0,
            ),
        )
    )
    cases.append(
        _rolling_generated_case(
            case_id="volgan-rolling-trained",
            model="volgan",
            variant="fixed-2022-source-selected-update",
            evidence=volgan_evidence,
            evidence_reason=volgan_reason,
            gate=volgan_gate,
            gate_path=generated.volgan_production_gate,
            gate_reason=volgan_gate_reason,
            dataset=dataset,
            output_root=output_root,
            checkpoint_root=checkpoint_root,
            config=config,
        )
    )
    cases.append(
        _review_gated_case(
            gate=diffusion_gate,
            gate_path=generated.diffusion_production_gate,
            gate_reason=diffusion_gate_reason,
            scope="daily-public",
            case_id="diffusion-ad-seq-vol-public-settings",
            model="diffusion-ad-seq-vol",
            measure="historical-P",
            variant="published-settings-full-rolling-fit",
            objective="mean-variance",
            risk_parameter=1.0,
            resource=ProductionResource.GPU,
            command=_daily_command(
                model="diffusion-ad-seq-vol",
                case_id="diffusion-ad-seq-vol-public-settings",
                dataset=dataset,
                output_root=output_root,
                checkpoint_root=checkpoint_root,
                config=config,
                device="cuda",
                objective="mean-variance",
                risk_parameter=1.0,
            ),
        )
    )
    cases.append(
        _rolling_generated_case(
            case_id="diffusion-ad-seq-vol-rolling-trained",
            model="diffusion-ad-seq-vol",
            variant="fixed-2022-source-selected-update",
            evidence=diffusion_evidence,
            evidence_reason=diffusion_reason,
            gate=diffusion_gate,
            gate_path=generated.diffusion_production_gate,
            gate_reason=diffusion_gate_reason,
            dataset=dataset,
            output_root=output_root,
            checkpoint_root=checkpoint_root,
            config=config,
        )
    )
    ft_public_id = "diffusion-ad-seq-vol-ft-public-settings"
    if diffusion_base is None:
        cases.append(
            _blocked_case(
                case_id=ft_public_id,
                model="diffusion-ad-seq-vol-ft",
                measure="historical-P",
                variant="published-settings-full-rolling-fit",
                objective="mean-variance",
                risk_parameter=1.0,
                resource=ProductionResource.GPU,
                reason=(
                    diffusion_base_reason
                    or "fine-tuned diffusion base checkpoint is not selected"
                ),
            )
        )
    else:
        cases.append(
            _review_gated_case(
                gate=diffusion_ft_gate,
                gate_path=generated.diffusion_ft_production_gate,
                gate_reason=diffusion_ft_gate_reason,
                scope="daily-public",
                base_evidence=diffusion_base,
                case_id=ft_public_id,
                model="diffusion-ad-seq-vol-ft",
                measure="historical-P",
                variant="published-settings-full-rolling-fit",
                objective="mean-variance",
                risk_parameter=1.0,
                resource=ProductionResource.GPU,
                command=_daily_command(
                    model="diffusion-ad-seq-vol-ft",
                    case_id=ft_public_id,
                    dataset=dataset,
                    output_root=output_root,
                    checkpoint_root=checkpoint_root,
                    config=config,
                    device="cuda",
                    objective="mean-variance",
                    risk_parameter=1.0,
                    base_checkpoint=diffusion_base.initializer_checkpoint,
                    base_checkpoint_sha256=(
                        diffusion_base.initializer_checkpoint_sha256
                    ),
                ),
            )
        )
    cases.append(
        _rolling_generated_case(
            case_id="diffusion-ad-seq-vol-ft-rolling-trained",
            model="diffusion-ad-seq-vol-ft",
            variant="fixed-2022-source-selected-update",
            evidence=diffusion_ft_evidence,
            evidence_reason=diffusion_ft_reason,
            gate=diffusion_ft_gate,
            gate_path=generated.diffusion_ft_production_gate,
            gate_reason=diffusion_ft_gate_reason,
            dataset=dataset,
            output_root=output_root,
            checkpoint_root=checkpoint_root,
            config=config,
        )
    )
    return tuple(cases)


def _rolling_generated_case(
    *,
    case_id: str,
    model: str,
    variant: str,
    evidence: GeneratedSelectionEvidence | None,
    evidence_reason: str | None,
    gate: GeneratedProductionGateEvidence | None,
    gate_path: Path | None,
    gate_reason: str | None,
    dataset: Path,
    output_root: Path,
    checkpoint_root: Path,
    config: DailyProductionMatrixConfig,
) -> DailyProductionCase:
    if evidence is None:
        return _blocked_case(
            case_id=case_id,
            model=model,
            measure="historical-P",
            variant=variant,
            objective="mean-variance",
            risk_parameter=1.0,
            resource=ProductionResource.GPU,
            reason=evidence_reason or "missing authenticated generated evidence",
        )
    return _review_gated_case(
        gate=gate,
        gate_path=gate_path,
        gate_reason=gate_reason,
        scope="daily-rolling",
        evidence=evidence,
        case_id=case_id,
        model=model,
        measure="historical-P",
        variant=variant,
        objective="mean-variance",
        risk_parameter=1.0,
        resource=ProductionResource.GPU,
        command=_daily_command(
            model=model,
            case_id=case_id,
            dataset=dataset,
            output_root=output_root,
            checkpoint_root=checkpoint_root,
            config=config,
            device="cuda",
            objective="mean-variance",
            risk_parameter=1.0,
            epochs=evidence.selected_epochs,
            model_label=evidence.initializer_model_label,
            warm_start_checkpoint=evidence.initializer_checkpoint,
            warm_start_checkpoint_sha256=(
                evidence.initializer_checkpoint_sha256
            ),
            base_checkpoint=evidence.base_checkpoint,
            base_checkpoint_sha256=evidence.base_checkpoint_sha256,
        ),
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


def _daily_command(
    *,
    model: str,
    case_id: str,
    dataset: Path,
    output_root: Path,
    checkpoint_root: Path,
    config: DailyProductionMatrixConfig,
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
) -> tuple[str, ...]:
    command = [
        "python",
        "-m",
        "quantdata.sdlv.daily.front_spot.hedging.runners.run_daily",
        "--model",
        model,
        "--dataset",
        str(dataset),
        "--validate-dataset",
        "--output",
        str(output_root / case_id),
        "--checkpoint-directory",
        str(checkpoint_root / case_id),
        "--observation-count",
        str(config.observation_count),
        "--position-cap",
        f"{config.position_cap:g}",
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
    return tuple(command)


def _objective_grids(
    config: DailyProductionMatrixConfig,
) -> tuple[tuple[str, tuple[float, ...]], ...]:
    return (
        ("mean-variance", config.mean_variance_parameters),
        ("mean-volatility", config.mean_volatility_parameters),
        ("entropy", config.entropy_parameters),
        ("power-exponential", config.power_exponential_parameters),
    )


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


def _review_gated_case(
    *,
    gate: GeneratedProductionGateEvidence | None,
    gate_path: Path | None,
    gate_reason: str | None,
    scope: str,
    evidence: GeneratedSelectionEvidence | None = None,
    base_evidence: GeneratedSelectionEvidence | None = None,
    **kwargs: object,
) -> DailyProductionCase:
    if gate is not None:
        try:
            require_generated_production_scope(
                gate,
                scope=scope,
                selection=evidence,
                base_initializer=base_evidence,
            )
        except ValueError as error:
            blocked = dict(kwargs)
            blocked.pop("command", None)
            return _blocked_case(
                reason=f"invalid generated production gate: {error}",
                **blocked,
            )
        return _ready_case(
            reviewed_commit=gate.reviewed_commit,
            production_gate_sha256=gate.gate_sha256,
            production_gate_path=str(Path(gate_path).resolve()),
            **kwargs,
        )
    reason = gate_reason or "awaiting generated-model production gate"
    if gate_path is not None:
        blocked = dict(kwargs)
        blocked.pop("command", None)
        return _blocked_case(reason=reason, **blocked)
    return _waiting_gate_case(reason=reason, **kwargs)


def _waiting_gate_case(*, reason: str, **kwargs: object) -> DailyProductionCase:
    return DailyProductionCase(
        readiness=ProductionReadiness.WAITING_GATE,
        blocked_reason=reason,
        **kwargs,
    )


def main() -> None:
    """Print or save the immutable production matrix without executing it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=default_public_candidate_path())
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
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
    config = DailyProductionMatrixConfig()
    cases = build_daily_production_matrix(
        dataset=arguments.dataset,
        output_root=arguments.output_root,
        checkpoint_root=arguments.checkpoint_root,
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
        "schema": "hedging-daily-production-matrix-v1",
        "code_version": str(type(config).version),
        "config": config.to_dict(),
        "generated_reviewed_commit": arguments.generated_reviewed_commit,
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
        "blocked_count": sum(
            case.readiness is ProductionReadiness.BLOCKED for case in cases
        ),
        "waiting_gate_count": sum(
            case.readiness is ProductionReadiness.WAITING_GATE for case in cases
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
    print(str(arguments.output), flush=True)


if __name__ == "__main__":
    main()
