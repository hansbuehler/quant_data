import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from quantdata.sdlv.daily.front_spot.hedging.runners import production_matrix
from quantdata.sdlv.daily.front_spot.hedging.runners.production_matrix import (
    DailyProductionMatrixConfig,
    GeneratedProductionInputs,
    GrommProductionInputs,
    ProductionReadiness,
    ProductionResource,
    build_daily_production_matrix,
)


def _matrix(
    *,
    dataset: Path = Path("/public/dataset"),
    gromm: GrommProductionInputs = GrommProductionInputs(),
    generated: GeneratedProductionInputs = GeneratedProductionInputs(),
):
    return build_daily_production_matrix(
        dataset=dataset,
        output_root=Path("/outputs"),
        checkpoint_root=Path("/checkpoints"),
        gromm=gromm,
        generated=generated,
    )


def test_daily_production_matrix_is_complete_unique_and_resource_partitioned() -> None:
    cases = _matrix()
    assert len(cases) == 53
    assert len({case.case_id for case in cases}) == len(cases)
    assert sum(case.model == "gromm" for case in cases) == 26
    assert all(
        case.resource is ProductionResource.GPU
        for case in cases
        if case.model in {"gromm", "volgan", "diffusion-ad-seq-vol", "diffusion-ad-seq-vol-ft"}
    )
    assert all(
        case.readiness is ProductionReadiness.BLOCKED
        for case in cases
        if case.model == "gromm"
    )


def test_native_commands_pin_common_daily_controls() -> None:
    config = DailyProductionMatrixConfig()
    cases = _matrix()
    native = tuple(
        case
        for case in cases
        if case.resource is ProductionResource.CPU
    )
    assert native
    for case in native:
        assert case.readiness is ProductionReadiness.READY
        assert case.command is not None
        assert case.command[0:3] == (
            "python",
            "-m",
            "quantdata.sdlv.daily.front_spot.hedging.runners.run_daily",
        )
        assert case.command[case.command.index("--observation-count") + 1] == str(
            config.observation_count
        )
        assert case.command[case.command.index("--position-cap") + 1] == "1000"
        assert case.command[case.command.index("--device") + 1] == "cpu"
        assert case.command[case.command.index("--start") + 1] == "2023-01-01"
        assert case.command[case.command.index("--end") + 1] == "2025-12-31"


@pytest.mark.parametrize(
    ("physical_accepted_count", "expected_p_readiness"),
    (
        (2, ProductionReadiness.BLOCKED),
        (3, ProductionReadiness.READY),
    ),
)
def test_gromm_readiness_uses_authenticated_q_and_physical_range_gates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    physical_accepted_count: int,
    expected_p_readiness: ProductionReadiness,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text("{}\n", encoding="utf-8")
    architecture = tmp_path / "architecture.json"
    architecture.write_text("{}\n", encoding="utf-8")
    summary_path = tmp_path / "range.json"
    summary_path.write_text("{}\n", encoding="utf-8")
    artifact_root = tmp_path / "artifacts"

    monkeypatch.setattr(
        production_matrix,
        "load_gromm_architecture_selection",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        production_matrix,
        "load_completed_rolling_gromm2_range",
        lambda *_args, **_kwargs: SimpleNamespace(
            evaluation_start=DailyProductionMatrixConfig().evaluation_start,
            evaluation_end=DailyProductionMatrixConfig().evaluation_end,
            eligible_origin_count=3,
            physical_accepted_count=physical_accepted_count,
        ),
    )
    cases = _matrix(
        dataset=dataset,
        gromm=GrommProductionInputs(
            market_artifact_root=artifact_root,
            architecture_selection=architecture,
            artifact_range_summary=summary_path,
        ),
    )
    gromm_cases = tuple(case for case in cases if case.model == "gromm")
    q_cases = tuple(case for case in gromm_cases if case.measure == "Q")
    p_cases = tuple(case for case in gromm_cases if case.measure == "P")
    assert all(case.readiness is ProductionReadiness.READY for case in q_cases)
    assert all(case.readiness is expected_p_readiness for case in p_cases)
    for case in q_cases:
        assert case.command is not None
        assert case.command[case.command.index("--measure") + 1] == "Q"
        assert case.command[case.command.index("--device") + 1] == "cuda"
        assert case.command[case.command.index("--market-artifact-root") + 1] == str(
            artifact_root
        )
        assert case.command[case.command.index("--architecture-selection") + 1] == str(
            architecture
        )
    if expected_p_readiness is ProductionReadiness.BLOCKED:
        assert all("2/3 origins" in case.blocked_reason for case in p_cases)


def test_generated_rolling_cases_require_selected_artifacts() -> None:
    missing = {case.case_id: case for case in _matrix()}
    assert (
        missing["volgan-rolling-trained"].readiness
        is ProductionReadiness.BLOCKED
    )
    assert "initializer" in missing["volgan-rolling-trained"].blocked_reason

    supplied = GeneratedProductionInputs(
        volgan_initializer=Path("/training/volgan-2022.pt"),
        volgan_update_epochs=7,
        diffusion_initializer=Path("/training/diffusion-2022.pt"),
        diffusion_update_epochs=11,
        diffusion_ft_base=Path("/training/diffusion-base.pt"),
        diffusion_ft_initializer=Path("/training/diffusion-ft-2022.pt"),
        diffusion_ft_update_epochs=5,
    )
    cases = {case.case_id: case for case in _matrix(generated=supplied)}
    for case_id, epochs in (
        ("volgan-rolling-trained", "7"),
        ("diffusion-ad-seq-vol-rolling-trained", "11"),
        ("diffusion-ad-seq-vol-ft-rolling-trained", "5"),
    ):
        case = cases[case_id]
        assert case.readiness is ProductionReadiness.WAITING_GATE
        assert case.command is not None
        assert case.command[case.command.index("--epochs") + 1] == epochs
        assert case.command[case.command.index("--device") + 1] == "cuda"


def test_production_matrix_cli_serializes_dates_and_generated_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "matrix.json"
    volgan = tmp_path / "volgan.pt"
    diffusion = tmp_path / "diffusion.pt"
    diffusion_base = tmp_path / "diffusion-base.pt"
    diffusion_ft = tmp_path / "diffusion-ft.pt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "production_matrix",
            "--dataset",
            str(tmp_path / "dataset"),
            "--output-root",
            str(tmp_path / "outputs"),
            "--checkpoint-root",
            str(tmp_path / "checkpoints"),
            "--output",
            str(output),
            "--volgan-initializer",
            str(volgan),
            "--volgan-update-epochs",
            "7",
            "--diffusion-initializer",
            str(diffusion),
            "--diffusion-update-epochs",
            "11",
            "--diffusion-ft-base",
            str(diffusion_base),
            "--diffusion-ft-initializer",
            str(diffusion_ft),
            "--diffusion-ft-update-epochs",
            "5",
        ],
    )

    production_matrix.main()

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["config"]["evaluation_start"] == "2023-01-01"
    assert payload["config"]["evaluation_end"] == "2025-12-31"
    case = next(
        value
        for value in payload["cases"]
        if value["case_id"] == "volgan-rolling-trained"
    )
    assert case["readiness"] == ProductionReadiness.WAITING_GATE.value
    assert case["command"][case["command"].index("--epochs") + 1] == "7"
    assert case["command"][
        case["command"].index("--warm-start-checkpoint") + 1
    ] == str(volgan)
    cases = {value["case_id"]: value for value in payload["cases"]}
    for case_id, initializer, epochs in (
        ("diffusion-ad-seq-vol-rolling-trained", diffusion, "11"),
        ("diffusion-ad-seq-vol-ft-rolling-trained", diffusion_ft, "5"),
    ):
        case = cases[case_id]
        assert case["readiness"] == ProductionReadiness.WAITING_GATE.value
        assert case["command"][case["command"].index("--epochs") + 1] == epochs
        assert case["command"][
            case["command"].index("--warm-start-checkpoint") + 1
        ] == str(initializer)
    fine_tuned = cases["diffusion-ad-seq-vol-ft-rolling-trained"]
    assert fine_tuned["command"][
        fine_tuned["command"].index("--base-checkpoint") + 1
    ] == str(diffusion_base)
