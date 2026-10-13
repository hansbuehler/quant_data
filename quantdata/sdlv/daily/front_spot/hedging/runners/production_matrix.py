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


class ProductionResource(StrEnum):
    """Primary execution resource for one production case."""

    CPU = "cpu"
    GPU = "gpu"


class ProductionReadiness(StrEnum):
    """Whether a case can be launched by the current daily runner."""

    READY = "ready"
    WAITING_GATE = "waiting-gate"
    BLOCKED = "blocked"


@version("1.1", dependencies=[DailyResultsConfig])
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
    """Selected immutable artifacts required by generated rolling cases."""

    volgan_initializer: Path | None = None
    volgan_update_epochs: int | None = None
    diffusion_initializer: Path | None = None
    diffusion_update_epochs: int | None = None
    diffusion_ft_base: Path | None = None
    diffusion_ft_initializer: Path | None = None
    diffusion_ft_update_epochs: int | None = None

    def __post_init__(self) -> None:
        for name, epochs in (
            ("volgan_update_epochs", self.volgan_update_epochs),
            ("diffusion_update_epochs", self.diffusion_update_epochs),
            ("diffusion_ft_update_epochs", self.diffusion_ft_update_epochs),
        ):
            verify_inp(
                epochs is None or epochs >= 1,
                lambda name=name: f"{name} must be positive when supplied",
            )


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
    missing = []
    if inputs.market_artifact_root is None:
        missing.append("complete rolling market-artifact root")
    if inputs.architecture_selection is None:
        missing.append("authenticated 2022 hedge-architecture selection")
    if inputs.artifact_range_summary is None:
        missing.append("completed rolling fit-range summary")
    range_summary = None
    invalid_reason = None
    if not missing:
        try:
            dataset_manifest_sha256 = file_sha256(Path(dataset) / "manifest.json")
            load_gromm_architecture_selection(
                inputs.architecture_selection,
                expected_dataset_manifest_sha256=dataset_manifest_sha256,
            )
            range_summary = load_completed_rolling_gromm2_range(
                inputs.artifact_range_summary,
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
            invalid_reason = f"invalid GROMM production evidence: {error}"
    common_reason = (
        "missing " + ", ".join(missing)
        if missing
        else invalid_reason
    )
    cases: list[DailyProductionCase] = []
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
                if common_reason is not None:
                    cases.append(_blocked_case(reason=common_reason, **common))
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
    cases: list[DailyProductionCase] = []
    cases.append(
        _waiting_gate_case(
            case_id="volgan-public-settings",
            model="volgan",
            measure="historical-P",
            variant="published-settings-full-rolling-fit",
            objective="published-cost-lasso",
            risk_parameter=1.0,
            resource=ProductionResource.GPU,
            reason="awaiting generated-model review and GPU-readiness gate",
            command=_daily_command(
                model="volgan",
                case_id="volgan-public-settings",
                dataset=dataset,
                output_root=output_root,
                checkpoint_root=checkpoint_root,
                config=config,
                device="cuda",
            ),
        )
    )
    cases.append(
        _rolling_generated_case(
            case_id="volgan-rolling-trained",
            model="volgan",
            variant="fixed-2022-source-selected-update",
            initializer=generated.volgan_initializer,
            update_epochs=generated.volgan_update_epochs,
            base_checkpoint=None,
            dataset=dataset,
            output_root=output_root,
            checkpoint_root=checkpoint_root,
            config=config,
        )
    )
    cases.append(
        _waiting_gate_case(
            case_id="diffusion-ad-seq-vol-public-settings",
            model="diffusion-ad-seq-vol",
            measure="historical-P",
            variant="published-settings-full-rolling-fit",
            objective="published-cost-lasso",
            risk_parameter=1.0,
            resource=ProductionResource.GPU,
            reason="awaiting generated-model review and GPU-readiness gate",
            command=_daily_command(
                model="diffusion-ad-seq-vol",
                case_id="diffusion-ad-seq-vol-public-settings",
                dataset=dataset,
                output_root=output_root,
                checkpoint_root=checkpoint_root,
                config=config,
                device="cuda",
            ),
        )
    )
    cases.append(
        _rolling_generated_case(
            case_id="diffusion-ad-seq-vol-rolling-trained",
            model="diffusion-ad-seq-vol",
            variant="fixed-2022-source-selected-update",
            initializer=generated.diffusion_initializer,
            update_epochs=generated.diffusion_update_epochs,
            base_checkpoint=None,
            dataset=dataset,
            output_root=output_root,
            checkpoint_root=checkpoint_root,
            config=config,
        )
    )
    ft_public_id = "diffusion-ad-seq-vol-ft-public-settings"
    if generated.diffusion_ft_base is None:
        cases.append(
            _blocked_case(
                case_id=ft_public_id,
                model="diffusion-ad-seq-vol-ft",
                measure="historical-P",
                variant="published-settings-full-rolling-fit",
                objective="published-cost-lasso",
                risk_parameter=1.0,
                resource=ProductionResource.GPU,
                reason="fine-tuned diffusion base checkpoint is not selected",
            )
        )
    else:
        cases.append(
            _waiting_gate_case(
                case_id=ft_public_id,
                model="diffusion-ad-seq-vol-ft",
                measure="historical-P",
                variant="published-settings-full-rolling-fit",
                objective="published-cost-lasso",
                risk_parameter=1.0,
                resource=ProductionResource.GPU,
                reason="awaiting generated-model review and GPU-readiness gate",
                command=_daily_command(
                    model="diffusion-ad-seq-vol-ft",
                    case_id=ft_public_id,
                    dataset=dataset,
                    output_root=output_root,
                    checkpoint_root=checkpoint_root,
                    config=config,
                    device="cuda",
                    base_checkpoint=generated.diffusion_ft_base,
                ),
            )
        )
    cases.append(
        _rolling_generated_case(
            case_id="diffusion-ad-seq-vol-ft-rolling-trained",
            model="diffusion-ad-seq-vol-ft",
            variant="fixed-2022-source-selected-update",
            initializer=generated.diffusion_ft_initializer,
            update_epochs=generated.diffusion_ft_update_epochs,
            base_checkpoint=generated.diffusion_ft_base,
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
    initializer: Path | None,
    update_epochs: int | None,
    base_checkpoint: Path | None,
    dataset: Path,
    output_root: Path,
    checkpoint_root: Path,
    config: DailyProductionMatrixConfig,
) -> DailyProductionCase:
    missing = []
    if initializer is None:
        missing.append("fixed pre-evaluation initializer")
    if update_epochs is None:
        missing.append("2022 one-standard-error update budget")
    if model.endswith("-ft") and base_checkpoint is None:
        missing.append("fine-tuning base checkpoint")
    if missing:
        return _blocked_case(
            case_id=case_id,
            model=model,
            measure="historical-P",
            variant=variant,
            objective="published-cost-lasso",
            risk_parameter=1.0,
            resource=ProductionResource.GPU,
            reason="missing " + ", ".join(missing),
        )
    return _waiting_gate_case(
        case_id=case_id,
        model=model,
        measure="historical-P",
        variant=variant,
        objective="published-cost-lasso",
        risk_parameter=1.0,
        resource=ProductionResource.GPU,
        reason="awaiting generated-model review and GPU-readiness gate",
        command=_daily_command(
            model=model,
            case_id=case_id,
            dataset=dataset,
            output_root=output_root,
            checkpoint_root=checkpoint_root,
            config=config,
            device="cuda",
            epochs=update_epochs,
            warm_start_checkpoint=initializer,
            base_checkpoint=base_checkpoint,
        ),
    )


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
    warm_start_checkpoint: Path | None = None,
    base_checkpoint: Path | None = None,
    measure: str | None = None,
    market_artifact_root: Path | None = None,
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
    if objective is not None:
        command.extend(("--objective", objective))
    if risk_parameter is not None:
        command.extend(("--risk-parameter", f"{risk_parameter:g}"))
    if epochs is not None:
        command.extend(("--epochs", str(epochs)))
    if warm_start_checkpoint is not None:
        command.extend(("--warm-start-checkpoint", str(warm_start_checkpoint)))
    if base_checkpoint is not None:
        command.extend(("--base-checkpoint", str(base_checkpoint)))
    if measure is not None:
        command.extend(("--measure", measure))
    if market_artifact_root is not None:
        command.extend(("--market-artifact-root", str(market_artifact_root)))
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
    parser.add_argument("--volgan-initializer", type=Path)
    parser.add_argument("--volgan-update-epochs", type=int)
    parser.add_argument("--diffusion-initializer", type=Path)
    parser.add_argument("--diffusion-update-epochs", type=int)
    parser.add_argument("--diffusion-ft-base", type=Path)
    parser.add_argument("--diffusion-ft-initializer", type=Path)
    parser.add_argument("--diffusion-ft-update-epochs", type=int)
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
            volgan_initializer=arguments.volgan_initializer,
            volgan_update_epochs=arguments.volgan_update_epochs,
            diffusion_initializer=arguments.diffusion_initializer,
            diffusion_update_epochs=arguments.diffusion_update_epochs,
            diffusion_ft_base=arguments.diffusion_ft_base,
            diffusion_ft_initializer=arguments.diffusion_ft_initializer,
            diffusion_ft_update_epochs=arguments.diffusion_ft_update_epochs,
        ),
    )
    payload = {
        "schema": "hedging-daily-production-matrix-v1",
        "code_version": str(type(config).version),
        "config": config.to_dict(),
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
